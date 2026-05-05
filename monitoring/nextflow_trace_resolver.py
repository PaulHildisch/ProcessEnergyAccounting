import csv
import os
import re
from dataclasses import dataclass

_DOCKER_CONTAINER_ID_RE = re.compile(
    r"/docker/([0-9a-f]{64})"  # cgroup v1: .../docker/<64-hex>
    r"|"
    r"/docker-([0-9a-f]{64})\.scope"  # cgroup v2: .../docker-<64-hex>.scope
)


@dataclass(frozen=True)
class NextflowTaskMetadata:
    workflow_run_id: str
    pipeline_name: str
    task_id: str
    task_name: str
    task_tag: str
    executor: str
    work_dir: str
    native_id: str
    group_id: str
    submit_time_ms: int | None
    start_time_ms: int | None
    complete_time_ms: int | None


class NextflowTraceResolver:
    TRACE_FIELDS = {
        "task_id": ("task_id", "task_id"),
        "task_name": ("process", "name"),
        "task_tag": ("tag", "tag"),
        "executor": ("executor", "executor"),
        "work_dir": ("workdir", "work_dir"),
        "native_id": ("native_id", "nativeId"),
    }

    def __init__(self, trace_path, workflow_run_id, pipeline_name="", refresh=True):
        self.trace_path = trace_path
        self.workflow_run_id = workflow_run_id
        self.pipeline_name = pipeline_name
        self.refresh = refresh
        self._trace_mtime_ns = None
        self._tasks = []
        self._tasks_by_native_pid = {}
        self._tasks_by_container_id = {}
        self._tasks_by_work_dir = {}

    def resolve_processes(self, processes, timestamp=None):
        self._refresh_if_needed()
        timestamp_ms = self._normalize_timestamp_ms(timestamp)

        pid_map = {process["pid"]: process for process in processes}
        assignments = {}

        for process in processes:
            task = self._match_process(process, timestamp_ms)
            if task is not None:
                assignments[process["pid"]] = task

        updated = True
        while updated:
            updated = False
            for process in processes:
                pid = process["pid"]
                if pid in assignments:
                    continue
                parent = process.get("ppid")
                if parent in assignments:
                    assignments[pid] = assignments[parent]
                    updated = True

        resolved = {}
        for pid, process in pid_map.items():
            task = assignments.get(pid)
            if task is not None:
                resolved[pid] = self._as_dict(task)
                continue

            fallback_root = self._find_root_pid(pid, pid_map)
            resolved[pid] = {
                "workflow_run_id": self.workflow_run_id,
                "pipeline_name": self.pipeline_name,
                "task_id": "",
                "task_name": "",
                "task_tag": "",
                "executor": "",
                "work_dir": "",
                "native_id": "",
                "group_id": f"{self.workflow_run_id}/pidtree/{fallback_root}",
            }

        return resolved

    def _refresh_if_needed(self):
        if not self.refresh and self._tasks:
            return
        try:
            mtime_ns = os.stat(self.trace_path).st_mtime_ns
        except OSError:
            self._clear_indexes()
            return

        if self._trace_mtime_ns == mtime_ns and self._tasks:
            return

        self._trace_mtime_ns = mtime_ns
        self._load_trace()

    def _load_trace(self):
        self._clear_indexes()
        with open(self.trace_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                task = self._build_task(row)
                if task is None:
                    continue
                self._tasks.append(task)
                native_pid = self._parse_native_pid(task.native_id)
                if native_pid is not None:
                    self._tasks_by_native_pid.setdefault(native_pid, []).append(task)
                container_id = self._parse_container_id_from_native_id(task.native_id)
                if container_id is not None:
                    self._tasks_by_container_id.setdefault(container_id, []).append(
                        task
                    )
                if task.work_dir:
                    self._tasks_by_work_dir.setdefault(task.work_dir, []).append(task)

    def _clear_indexes(self):
        self._tasks = []
        self._tasks_by_native_pid = {}
        self._tasks_by_container_id = {}
        self._tasks_by_work_dir = {}

    def _build_task(self, row):
        task_id = self._get_first(row, *self.TRACE_FIELDS["task_id"]).strip()
        if not task_id:
            return None

        workflow_run_id = (
            self._get_first(row, "workflow_run_id").strip() or self.workflow_run_id
        )
        pipeline_name = (
            self._get_first(row, "pipeline_name").strip() or self.pipeline_name
        )
        task_name = self._get_first(row, *self.TRACE_FIELDS["task_name"]).strip()
        task_tag = self._get_first(row, *self.TRACE_FIELDS["task_tag"]).strip()
        executor = self._get_first(row, *self.TRACE_FIELDS["executor"]).strip()
        work_dir = self._normalize_path(
            self._get_first(row, *self.TRACE_FIELDS["work_dir"]).strip()
        )
        native_id = self._get_first(row, *self.TRACE_FIELDS["native_id"]).strip()
        submit_time_ms = self._parse_trace_timestamp(self._get_first(row, "submit"))
        start_time_ms = self._parse_trace_timestamp(self._get_first(row, "start"))
        complete_time_ms = self._parse_trace_timestamp(self._get_first(row, "complete"))

        return NextflowTaskMetadata(
            workflow_run_id=workflow_run_id,
            pipeline_name=pipeline_name,
            task_id=task_id,
            task_name=task_name,
            task_tag=task_tag,
            executor=executor,
            work_dir=work_dir,
            native_id=native_id,
            group_id=f"{workflow_run_id}/{task_id}",
            submit_time_ms=submit_time_ms,
            start_time_ms=start_time_ms,
            complete_time_ms=complete_time_ms,
        )

    def _match_process(self, process, timestamp_ms):
        task = self._pick_candidate(
            self._tasks_by_native_pid.get(process["pid"], []),
            timestamp_ms,
        )
        if task is not None:
            return task

        cgroup = process.get("cgroup") or ""
        if cgroup:
            cid = self._parse_container_id_from_cgroup(cgroup)
            if cid is not None:
                task = self._pick_candidate(
                    self._tasks_by_container_id.get(cid, []),
                    timestamp_ms,
                )
                if task is not None:
                    return task

        cwd = self._normalize_path(process.get("cwd"))
        if cwd:
            task = self._lookup_by_work_dir(cwd, timestamp_ms)
            if task is not None:
                return task

        cmdline = process.get("cmdline") or ""
        for work_dir, candidates in self._tasks_by_work_dir.items():
            if work_dir and work_dir in cmdline:
                task = self._pick_candidate(candidates, timestamp_ms)
                if task is not None:
                    return task

        return None

    def _lookup_by_work_dir(self, cwd, timestamp_ms):
        current = cwd
        while current and current != "/":
            task = self._pick_candidate(
                self._tasks_by_work_dir.get(current, []),
                timestamp_ms,
            )
            if task is not None:
                return task
            current = os.path.dirname(current)
        return self._pick_candidate(
            self._tasks_by_work_dir.get("/", []),
            timestamp_ms,
        )

    def _find_root_pid(self, pid, pid_map):
        current_pid = pid
        visited = set()
        while current_pid in pid_map and current_pid not in visited:
            visited.add(current_pid)
            parent_pid = pid_map[current_pid].get("ppid")
            if parent_pid in (None, 0) or parent_pid not in pid_map:
                return current_pid
            current_pid = parent_pid
        return pid

    def _as_dict(self, task):
        return {
            "workflow_run_id": task.workflow_run_id,
            "pipeline_name": task.pipeline_name,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_tag": task.task_tag,
            "executor": task.executor,
            "work_dir": task.work_dir,
            "native_id": task.native_id,
            "group_id": task.group_id,
        }

    def _get_first(self, row, *keys):
        for key in keys:
            value = row.get(key)
            if value is not None:
                return value
        return ""

    def _parse_native_pid(self, native_id):
        if not native_id:
            return None
        head = native_id.split("/")[0].split(";")[0].strip()
        if head.isdigit():
            return int(head)
        return None

    def _parse_container_id_from_native_id(self, native_id):
        if not native_id:
            return None
        head = native_id.split("/")[0].split(";")[0].strip()
        if len(head) not in (12, 64):
            return None
        if not all(c in "0123456789abcdefABCDEF" for c in head):
            return None
        return head[:12].lower()

    def _parse_container_id_from_cgroup(self, cgroup):
        if not cgroup:
            return None
        match = _DOCKER_CONTAINER_ID_RE.search(cgroup)
        if match:
            container_id = match.group(1) or match.group(2)
            return container_id[:12].lower()
        return None

    def _parse_trace_timestamp(self, value):
        if not value:
            return None
        head = value.strip().split(".")[0]
        if head.isdigit():
            return int(head)
        return None

    def _normalize_timestamp_ms(self, timestamp):
        if timestamp is None:
            return None
        if hasattr(timestamp, "value"):
            return int(timestamp.value // 1_000_000)
        return None

    def _pick_candidate(self, candidates, timestamp_ms):
        if not candidates:
            return None
        if timestamp_ms is None:
            return candidates[0]

        active = [
            task
            for task in candidates
            if self._task_matches_timestamp(task, timestamp_ms)
        ]
        if not active:
            return None
        if len(active) == 1:
            return active[0]
        return min(active, key=lambda task: self._task_distance(task, timestamp_ms))

    def _task_matches_timestamp(self, task, timestamp_ms):
        start_ms = task.start_time_ms or task.submit_time_ms
        complete_ms = task.complete_time_ms
        if start_ms is not None and timestamp_ms < start_ms:
            return False
        if complete_ms is not None and timestamp_ms > complete_ms:
            return False
        return True

    def _task_distance(self, task, timestamp_ms):
        start_ms = task.start_time_ms or task.submit_time_ms or timestamp_ms
        complete_ms = task.complete_time_ms or timestamp_ms
        if start_ms <= timestamp_ms <= complete_ms:
            return min(timestamp_ms - start_ms, complete_ms - timestamp_ms)
        return abs(timestamp_ms - start_ms)

    def _normalize_path(self, path_value):
        if not path_value:
            return ""
        return os.path.normpath(path_value)
