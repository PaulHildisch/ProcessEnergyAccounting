import logging
import os
import random
import re
import threading
from typing import Any, cast

import pandas as pd
from accumulation.cgroups import CgroupV2
from kubernetes import client, config, watch
from tabulate import tabulate

PERF_COUNTER_KEYS = [
    "delta_instructions",
    "delta_cycles",
    "delta_branch_instructions",
    "delta_cache_references",
    "delta_cache_misses",
    "delta_stalled_cycles_backend",
    "delta_llc_load_misses",
    "delta_llc_store_misses",
    "delta_cpu_migrations",
    "delta_page_faults_min",
    "delta_page_faults_maj",
    "delta_stalled_cycles_frontend",
    "delta_branch_misses",
    "delta_ref_cpu_cycles",
    "delta_l1d_load_misses",
    "delta_dtlb_load_misses",
    "delta_dtlb_store_misses",
    "delta_node_load_misses",
    "delta_fp_scalar",
    "delta_fp_128b_packed",
    "delta_fp_256b_packed",
    "delta_fp_512b_packed",
    "delta_fp_add_sub",
    "delta_fp_mult",
    "delta_fp_div",
    "delta_fp_mac",
]


class K8sManager:
    def __init__(self, kubeconfig, cgroups: CgroupV2, use_pod_regex: bool = False):
        self.kubeconfig_path = kubeconfig
        config.load_kube_config(kubeconfig)
        self.v1 = client.CoreV1Api()
        self.watcher = watch.Watch()
        self.namespace = "default"
        self.local_node_name = os.getenv("K8S_NODE_NAME") or os.getenv("NODE_NAME")
        self.use_pod_regex = use_pod_regex
        self.pod_name_regex = ""
        self.pod_name_pattern = None
        if self.use_pod_regex:
            self.pod_name_regex = (os.getenv("K8S_POD_REGEX") or "").strip()
            if self.pod_name_regex:
                try:
                    self.pod_name_pattern = re.compile(self.pod_name_regex)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid K8S_POD_REGEX '{self.pod_name_regex}': {exc}"
                    ) from exc
        self.logged_started_containers = set()
        self.logged_terminated_containers = set()
        self.pod_container_to_pids_to_metrics = {}
        self.pod_container_to_pids_to_metrics_summed = {}
        self.cgroups = cgroups

    def run(self, callback=None):
        logging.info("Starting Kubernetes event watcher threads...")
        logging.info("Kubeconfig path: %s", self.kubeconfig_path or "default")
        logging.info("Watching namespace: %s", self.namespace)
        logging.info(
            "Filtering pod events for local node: %s",
            self.local_node_name or "<disabled>",
        )
        logging.info(
            "Filtering pod events by regex: %s",
            self.pod_name_regex if self.use_pod_regex else "<disabled via CLI>",
        )
        start_thread = threading.Thread(
            target=self.get_pod_container_start_events, args=(callback,), daemon=True
        )
        die_thread = threading.Thread(
            target=self.get_pod_container_die_events, args=(callback,), daemon=True
        )
        start_thread.start()
        die_thread.start()

    def _matches_pod_name(self, pod_name: str) -> bool:
        if not self.pod_name_pattern:
            return True
        return self.pod_name_pattern.search(pod_name) is not None

    def _is_local_pod(self, pod) -> bool:
        pod_name = getattr(pod.metadata, "name", "")
        if not self._matches_pod_name(pod_name):
            logging.debug(
                "Ignoring pod event due to regex filter: pod=%s regex=%s",
                pod_name,
                self.pod_name_regex,
            )
            return False

        if not self.local_node_name:
            return True
        pod_node_name = getattr(pod.spec, "node_name", None)
        if pod_node_name != self.local_node_name:
            logging.debug(
                "Ignoring non-local pod event: pod=%s namespace=%s node=%s local_node=%s",
                pod.metadata.name,
                pod.metadata.namespace,
                pod_node_name,
                self.local_node_name,
            )
            return False
        return True

    def get_pod_container_start_events(self, callback=None):
        for event in self.watcher.stream(
            self.v1.list_namespaced_pod, namespace=self.namespace
        ):
            event_dict = cast(dict[str, Any], event)
            if event_dict["type"] in ("ADDED", "MODIFIED"):
                pod = event_dict["object"]
                if not self._is_local_pod(pod):
                    continue
                container_statuses = pod.status.container_statuses or []
                for container in pod.spec.containers:
                    container_status = next(
                        (cs for cs in container_statuses if cs.name == container.name),
                        None,
                    )
                    if (
                        container_status
                        and container_status.container_id
                        and container_status.state
                        and container_status.state.running is not None
                    ):
                        raw_id = container_status.container_id
                        container_id = raw_id.split("://", 1)[-1]
                        pod_id = pod.metadata.uid
                        pod_name = pod.metadata.name
                        namespace = pod.metadata.namespace
                        if container_id in self.logged_started_containers:
                            continue
                        self.logged_started_containers.add(container_id)
                        logging.info(
                            "[POD STARTED] pod=%s container=%s namespace=%s pod_id=%s container_id=%s",
                            pod_name,
                            container.name,
                            namespace,
                            pod_id,
                            container_id,
                        )
                        if not hasattr(self, "pod_to_container_id_dict"):
                            self.pod_to_container_id_dict = {}
                        self.pod_to_container_id_dict[pod_id] = container_id
                        if callback:
                            callback(
                                container_id,
                                "start",
                                pod_name,
                                pod_id,
                                container.name,
                            )
                    else:
                        logging.debug(
                            "POD START EVENT IGNORED: pod=%s container=%s has no running state yet",
                            pod.metadata.name,
                            container.name,
                        )

    def get_pod_container_die_events(self, callback=None):
        for event in self.watcher.stream(
            self.v1.list_namespaced_pod, namespace=self.namespace
        ):
            event_dict = cast(dict[str, Any], event)
            if event_dict["type"] == "MODIFIED":
                pod = event_dict["object"]
                if not self._is_local_pod(pod):
                    continue
                container_statuses = pod.status.container_statuses or []
                for container in pod.spec.containers:
                    container_status = next(
                        (cs for cs in container_statuses if cs.name == container.name),
                        None,
                    )
                    if (
                        container_status
                        and container_status.container_id
                        and container_status.state
                        and container_status.state.terminated is not None
                        and container_status.state.terminated.reason == "Completed"
                    ):
                        raw_id = container_status.container_id
                        container_id = raw_id.split("://", 1)[-1]
                        pod_id = pod.metadata.uid
                        pod_name = pod.metadata.name
                        namespace = pod.metadata.namespace
                        termination = container_status.state.terminated
                        if container_id in self.logged_terminated_containers:
                            continue
                        self.logged_terminated_containers.add(container_id)
                        logging.info(
                            "[POD TERMINATED] pod=%s container=%s namespace=%s pod_id=%s container_id=%s reason=%s exit_code=%s finished_at=%s",
                            pod_name,
                            container.name,
                            namespace,
                            pod_id,
                            container_id,
                            termination.reason,
                            termination.exit_code,
                            termination.finished_at,
                        )
                        if not hasattr(self, "pod_to_container_id_dict"):
                            self.pod_to_container_id_dict = {}
                        self.pod_to_container_id_dict[pod_id] = container_id
                        if callback:
                            callback(
                                container_id,
                                "die",
                                pod_name,
                                pod_id,
                                container.name,
                            )
                    else:
                        logging.debug(
                            "POD MODIFY EVENT IGNORED: pod=%s container=%s is not successfully completed",
                            pod.metadata.name,
                            container.name,
                        )

    def get_latest_pod_container_to_pid_mapping(self, pid_callback=None):
        if pid_callback:
            print("Merging container events with PID and metrics updates...")
        return self.cgroups.get_container_names_to_pids()

    def merge_pod_containers_with_pids_from_deltas(self, deltas):
        pod_to_pids = self.cgroups.get_container_names_to_pids()
        if not pod_to_pids:
            logging.info("No pods or PIDs found.")
            return {}

        summed_by_pod = {}
        for pod_name, pids in pod_to_pids.items():
            if not self._matches_pod_name(pod_name):
                logging.debug(
                    "Skipping aggregated pod due to regex filter: pod=%s regex=%s",
                    pod_name,
                    self.pod_name_regex,
                )
                continue

            matching_pids = [pid for pid in pids if pid in deltas]
            missing_pids = [pid for pid in pids if pid not in deltas]

            self.get_container_deltas_summed(pod_name, matching_pids, deltas)
            self.get_container_pids_deltas(pod_name, matching_pids, deltas)
            summed = self.pod_container_to_pids_to_metrics_summed[pod_name]
            summed_by_pod[pod_name] = summed

            logging.info(f"Pod: {pod_name}")
            logging.info(
                f"  PID count: {len(pids)} | Metrics found for: {len(matching_pids)} | Missing: {len(missing_pids)}"
            )
            if missing_pids:
                logging.debug(f"  Missing PIDs: {missing_pids}")

            if matching_pids:
                per_pid_metrics = []
                for pid in matching_pids:
                    m = deltas[pid]
                    per_pid_metrics.append(
                        {
                            "PID": pid,
                            "Name": m.get("name", ""),
                            "CPU(ns)": m.get("delta_cpu_ns", 0),
                            "IO(bytes)": m.get("delta_io_bytes", 0),
                            "Net(bytes)": m.get("delta_net_send_bytes", 0),
                            "Syscalls": m.get("syscall_count", 0),
                            "Instr": m.get("delta_instructions", 0),
                            "Cycles": m.get("delta_cycles", 0),
                            "BranchInstr": m.get("delta_branch_instructions", 0),
                            "CacheMiss": m.get("delta_cache_misses", 0),
                            "BranchMiss": m.get("delta_branch_misses", 0),
                            "L1DMiss": m.get("delta_l1d_load_misses", 0),
                            "LLCLoadMiss": m.get("delta_llc_load_misses", 0),
                            "RefCycles": m.get("delta_ref_cpu_cycles", 0),
                        }
                    )
                logging.info(
                    "\n" + tabulate(per_pid_metrics, headers="keys", tablefmt="github")
                )

                logging.info("  [SUMMED METRICS]")
                for k, v in summed.items():
                    logging.info(f"    {k:20}: {v}")
                self._log_perf_counter_summary(summed)

                validation = {}
                for k in summed.keys():
                    if k in ("pid", "ppid", "name"):
                        continue
                    pid_sum = sum(deltas[pid].get(k, 0) for pid in matching_pids)
                    validation[k] = (
                        pid_sum,
                        summed[k],
                        "OK" if pid_sum == summed[k] else "MISMATCH",
                    )
            else:
                logging.info("  No matching PIDs with metrics for this pod.")

        return summed_by_pod

    def get_container_pids_deltas(self, container, matching_pids, deltas):
        if container not in self.pod_container_to_pids_to_metrics:
            self.pod_container_to_pids_to_metrics[container] = {}
        for pid in matching_pids:
            self.pod_container_to_pids_to_metrics[container][pid] = deltas[pid]
        # Show example metrics for one pid of this container at debug level
        if matching_pids:
            example_pid = random.choice(matching_pids)
            example_metrics = deltas[example_pid]
            logging.debug(
                f"  Example metrics for container '{container}', pid {example_pid}: {example_metrics}"
            )
        return self.pod_container_to_pids_to_metrics

    def get_container_deltas_summed(self, container, matching_pids, deltas):
        exclude_keys = {"pid", "ppid", "name"}
        metrics_list = []
        for pid in matching_pids:
            if pid in deltas:
                filtered = {
                    k: v for k, v in deltas[pid].items() if k not in exclude_keys
                }

                # Flatten nested deltas so pod-level sums keep these features.
                syscall_class_deltas = filtered.pop("syscall_class_deltas", {}) or {}
                if isinstance(syscall_class_deltas, dict):
                    for cls, count in syscall_class_deltas.items():
                        filtered[f"syscall_class_{cls}"] = count

                fp_op_deltas = filtered.pop("fp_op_deltas", {}) or {}
                if isinstance(fp_op_deltas, dict):
                    for name, count in fp_op_deltas.items():
                        filtered[f"delta_{name}"] = count

                # Add delta_* aliases used by model artifacts.
                if "instructions" in filtered:
                    filtered.setdefault("delta_instructions", filtered["instructions"])
                if "cycles" in filtered:
                    filtered.setdefault("delta_cycles", filtered["cycles"])
                if "branch_instructions" in filtered:
                    filtered.setdefault(
                        "delta_branch_instructions", filtered["branch_instructions"]
                    )
                if "cache_misses" in filtered:
                    filtered.setdefault("delta_cache_misses", filtered["cache_misses"])

                metrics_list.append(filtered)
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            summed_metrics = df.sum(numeric_only=True).to_dict()
        else:
            summed_metrics = {}
        self.pod_container_to_pids_to_metrics_summed[container] = summed_metrics
        return self.pod_container_to_pids_to_metrics_summed

    @staticmethod
    def _log_perf_counter_summary(summed):
        logging.info("  [PERF COUNTERS]")
        for key in PERF_COUNTER_KEYS:
            logging.info(f"    {key:32}: {summed.get(key, 0)}")
