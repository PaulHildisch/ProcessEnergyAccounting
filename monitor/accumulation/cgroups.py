import logging
import os
import pprint
import re
import threading
import time

from cgroupspy import trees


class CgroupV2:
    def __init__(self, pid_map_callback=None):
        self.pids = set()
        self.path_to_pids = {}
        self.lock = threading.Lock()
        self.container_names_to_pids = {}
        self.pid_count_per_container = {}
        self.pid_map_callback = pid_map_callback

    def get_container_names_to_pids(self):
        """Return the current container_names_to_pids mapping."""
        return self.container_names_to_pids

    # def run(self, monitor_active):
    #     while monitor_active:
    #         # Continously search for new processes spawning in containers
    #         threading.Thread(target=self.monitor_pid_updates).start()
    #         time.sleep(1)  # Adjust the sleep time as needed

    # Docker event handler: called from DockerManager for Docker container lifecycle events.
    def handle_container_event(self, container_id, event_type, container_name):
        if event_type == "start":
            print(f"Container event: {event_type} for ({container_name})")
            # Find the cgroup path for this container
            target_pids = set(self.container_names_to_pids.get(container_name, []))
            cgroup_path = None
            for path, pids in self.path_to_pids.items():
                if set(pids) == target_pids and target_pids:
                    cgroup_path = path
                    break
            if cgroup_path:
                threading.Thread(
                    target=self.monitor_new_pids_for_container,
                    args=(container_name, cgroup_path),
                    daemon=True,
                ).start()

            # Match cgroup and init mapping from containers to PIDs
            self.path_to_pids = self.find_docker_cgroups_with_pids()
            self.match_containers_with_pids(
                container_id, container_name, self.path_to_pids
            )
            # Find the cgroup path for this container
            target_pids = set(self.container_names_to_pids.get(container_name, []))
            cgroup_path = None
            for path, pids in self.path_to_pids.items():
                if set(pids) == target_pids and target_pids:
                    cgroup_path = path
                    break
            if cgroup_path:
                threading.Thread(
                    target=self.monitor_new_pids_for_container,
                    args=(container_name, cgroup_path),
                    daemon=True,
                ).start()
            else:
                print(f"Could not find cgroup path for {container_name}")
        elif event_type == "die":
            print(f"Container event: {event_type} for ({container_name} with ID)")
            print(f"Removing stopped container {container_name} with from mapping")
            try:
                del self.container_names_to_pids[container_name]
            except KeyError:
                print(f"Container {container_name} not found in mapping during removal")

    # Kubernetes pod/container event handler.
    # Uses container_id + pod_uid for cgroup discovery/validation and stores PID mappings by pod_name.
    def handle_pod_container_event(
        self, container_id, event_type, pod_name, pod_uid, container_name
    ):
        if event_type == "start":
            logging.info(
                "[K8S CGROUP DISCOVERY] event=%s pod=%s container=%s pod_uid=%s container_id=%s",
                event_type,
                pod_name,
                container_name,
                pod_uid,
                container_id,
            )
            target_pids = set(self.container_names_to_pids.get(pod_name, []))
            cgroup_path = None
            for path, pids in self.path_to_pids.items():
                if set(pids) == target_pids and target_pids:
                    cgroup_path = path
                    break
            if cgroup_path:
                logging.info(
                    "[K8S PID MONITOR] Reusing existing cgroup path for pod=%s path=%s",
                    pod_name,
                    cgroup_path,
                )
                threading.Thread(
                    target=self.monitor_new_pids_for_container,
                    args=(pod_name, cgroup_path),
                    daemon=True,
                ).start()

            self.path_to_pids = self.find_k8s_cgroups_with_pids()
            self.match_k8s_containers_with_pids(
                container_id, pod_uid, pod_name, self.path_to_pids
            )
            target_pids = set(self.container_names_to_pids.get(pod_name, []))
            cgroup_path = None
            for path, pids in self.path_to_pids.items():
                if set(pids) == target_pids and target_pids:
                    cgroup_path = path
                    break
            if cgroup_path:
                logging.info(
                    "[K8S PID MONITOR] Starting PID monitor for pod=%s container=%s path=%s pid_count=%s",
                    pod_name,
                    container_name,
                    cgroup_path,
                    len(self.container_names_to_pids.get(pod_name, [])),
                )
                threading.Thread(
                    target=self.monitor_new_pids_for_container,
                    args=(pod_name, cgroup_path),
                    daemon=True,
                ).start()
            else:
                logging.warning(
                    "[K8S CGROUP DISCOVERY] No cgroup path found for pod=%s container=%s pod_uid=%s container_id=%s",
                    pod_name,
                    container_name,
                    pod_uid,
                    container_id,
                )
        elif event_type == "die":
            logging.info(
                "[K8S CGROUP DISCOVERY] event=%s pod=%s container=%s pod_uid=%s container_id=%s",
                event_type,
                pod_name,
                container_name,
                pod_uid,
                container_id,
            )
            self.remove_tracked_name(pod_name)

    # Docker cgroup discovery entrypoint.
    def find_docker_cgroups_with_pids(self):
        cgroup_paths = self._find_cgroup_paths()
        return self._get_pids_for_cgroup_paths(cgroup_paths)

    def find_k8s_cgroups_with_pids(self):
        cgroup_paths = self._find_k8s_cgroup_paths()
        logging.info(
            "[K8S CGROUP DISCOVERY] Found %s candidate cgroup paths",
            len(cgroup_paths),
        )
        return self._get_pids_for_cgroup_paths(cgroup_paths)

    # Docker-specific cgroup path scan.
    # Looks for systemd cgroup names like docker-<64hex>.scope.
    def _find_cgroup_paths(self):
        t = trees.Tree()
        pattern = re.compile(r"docker-[0-9a-f]{64}\.scope")
        cgroup_paths = []
        for node in t.root.walk():
            if pattern.search(node.name.decode()):
                try:
                    full_path_str = node.full_path.decode()
                    cgroup_paths.append(full_path_str)
                except Exception as e:
                    print(
                        f"Found inactive container without cgroup path...ignoring. Error: {e}"
                    )
        return cgroup_paths

    # Kubernetes-specific cgroup path scan for systemd/containerd-based pod cgroups.
    # Matches paths like:
    # kubepods.slice/kubepods-besteffort.slice/
    # kubepods-besteffort-pod<uid>.slice/cri-containerd-<container-id>.scope
    def _find_k8s_cgroup_paths(self):
        t = trees.Tree()
        pattern = re.compile(
            r"kubepods\.slice/kubepods-[^/]+\.slice/kubepods-[^/]*pod[0-9a-f_]+\.slice/cri-containerd-[0-9a-f]{64}\.scope$"
        )
        logging.debug(
            "[K8S CGROUP DISCOVERY] Scanning cgroup tree for kubepods/containerd paths"
        )
        cgroup_paths = []
        for node in t.root.walk():
            try:
                node_name = node.name.decode()
                full_path_str = node.full_path.decode()
                if "cri-containerd-" in node_name or "kubepods" in full_path_str:
                    logging.debug(
                        "[K8S CGROUP DISCOVERY] Inspecting node_name=%s full_path=%s",
                        node_name,
                        full_path_str,
                    )
                if pattern.search(full_path_str):
                    logging.info(
                        "[K8S CGROUP DISCOVERY] Regex matched node_name=%s full_path=%s",
                        node_name,
                        full_path_str,
                    )
                    cgroup_paths.append(full_path_str)
            except Exception as e:
                print(
                    f"Found inactive container without cgroup path...ignoring. Error: {e}"
                )
        return cgroup_paths

    # Generic PID extraction helper used by the Docker discovery flow.
    def _get_pids_for_cgroup_paths(self, cgroup_paths):
        for full_path_str in cgroup_paths:
            try:
                with open(full_path_str + "/cgroup.procs") as f:
                    pids = [int(x) for x in f.read().split()]
                    self.pids.update(pids)
                    with self.lock:
                        if full_path_str not in self.path_to_pids:
                            self.path_to_pids[full_path_str] = pids
            except Exception as e:
                print(f"Error reading PIDs for {full_path_str}: {e}")
        return self.path_to_pids

    # Shared matcher used by the Docker event flow.
    # It links a runtime/container ID to the discovered cgroup PID list.
    def match_containers_with_pids(self, container_id, container_name, path_to_pids):
        for path, pids in self.path_to_pids.items():
            if container_id in path:
                self.container_names_to_pids[container_name] = pids
                # print(f"Container | Initial PIDs: {self.container_names_to_pids}")
                if self.pid_map_callback:
                    self.pid_map_callback(self.container_names_to_pids)
                return self.container_names_to_pids

    # Kubernetes matcher: validates the path using both container_id and pod_uid,
    # but stores the resolved PID list under pod_name for higher-level task aggregation.
    def match_k8s_containers_with_pids(
        self, container_id, pod_uid, pod_name, path_to_pids
    ):
        normalized_pod_uid = pod_uid.replace("-", "_")
        for path, pids in self.path_to_pids.items():
            if container_id in path and normalized_pod_uid in path:
                self.container_names_to_pids[pod_name] = pids
                logging.info(
                    "[K8S PID MAP] pod=%s pod_uid=%s container_id=%s matched_cgroup=%s pid_count=%s",
                    pod_name,
                    pod_uid,
                    container_id,
                    path,
                    len(pids),
                )
                if self.pid_map_callback:
                    self.pid_map_callback(self.container_names_to_pids)
                return self.container_names_to_pids
        logging.warning(
            "[K8S PID MAP] No PID match found for pod=%s pod_uid=%s container_id=%s",
            pod_name,
            pod_uid,
            container_id,
        )

    def remove_tracked_name(self, tracked_name):
        logging.info(
            "[CGROUP PID MAP] Removing tracked name=%s from mapping", tracked_name
        )
        with self.lock:
            self.container_names_to_pids.pop(tracked_name, None)
            stale_paths = [
                path
                for path, pids in self.path_to_pids.items()
                if set(pids) == set()
                or self.container_names_to_pids.get(tracked_name) == pids
            ]
            for path in stale_paths:
                self.path_to_pids.pop(path, None)
            if self.pid_map_callback:
                self.pid_map_callback(self.container_names_to_pids)

    # Shared background monitor for both Docker containers and Kubernetes pod containers
    # once a matching cgroup path has been found.
    def monitor_new_pids_for_container(
        self, container_name, cgroup_path, poll_interval=1
    ):
        """Monitor the cgroup.procs file for pids and report them."""
        procs_file = os.path.join(cgroup_path, "cgroup.procs")
        seen_pids = set()
        read_failures = 0
        while True:
            try:
                with open(procs_file) as f:
                    current_pids = set(int(x) for x in f.read().split())
                read_failures = 0
                if not current_pids:
                    logging.info(
                        "[CGROUP PID UPDATE] name=%s cgroup=%s has no remaining PIDs; removing mapping",
                        container_name,
                        cgroup_path,
                    )
                    self.remove_tracked_name(container_name)
                    self.path_to_pids.pop(cgroup_path, None)
                    return
                new_pids = current_pids - seen_pids
                if new_pids:
                    logging.info(
                        "[CGROUP PID UPDATE] name=%s new_pids=%s total_pids=%s cgroup=%s",
                        container_name,
                        sorted(new_pids),
                        len(current_pids),
                        cgroup_path,
                    )
                with self.lock:
                    self.container_names_to_pids[container_name] = list(current_pids)
                    self.path_to_pids[cgroup_path] = list(current_pids)
                    if self.pid_map_callback:
                        self.pid_map_callback(self.container_names_to_pids)
                seen_pids = current_pids
            except Exception:
                read_failures += 1
                if read_failures >= 3:
                    logging.info(
                        "[CGROUP PID UPDATE] name=%s cgroup=%s became unavailable; removing mapping",
                        container_name,
                        cgroup_path,
                    )
                    self.remove_tracked_name(container_name)
                    self.path_to_pids.pop(cgroup_path, None)
                    return
            time.sleep(poll_interval)
