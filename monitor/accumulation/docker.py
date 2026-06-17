import logging
import random
import threading

import docker
import pandas as pd
from tabulate import tabulate

from accumulation.cgroups import CgroupV2


class DockerManager:
    def __init__(self, cgroups: CgroupV2):
        self.client = docker.from_env()
        self.docker_container_to_pids_to_metrics = {}
        self.docker_container_to_pids_to_metrics_summed = {}
        self.cgroups = cgroups
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

    def run(self, callback=None):
        start_thread = threading.Thread(
            target=self.get_container_start_events, args=(callback,), daemon=True
        )
        die_thread = threading.Thread(
            target=self.get_container_die_events, args=(callback,), daemon=True
        )
        start_thread.start()
        die_thread.start()

    def get_container_start_events(self, callback=None):
        events = self.client.events(decode=True, filters={"event": "start"})
        start_events = []
        for event in events:
            if event.get("Type") == "container" and event.get("Action") == "start":
                container_id = None
                container_name = None
                if "Actor" in event:
                    if "Attributes" in event["Actor"]:
                        container_name = event["Actor"]["Attributes"].get("name")
                    container_id = event["Actor"].get("ID")
                if callback and container_id:
                    callback(container_id, "start", container_name)
                start_events.append(event)
        return start_events

    def get_container_die_events(self, callback=None):
        events = self.client.events(decode=True, filters={"event": "die"})
        die_events = []
        for event in events:
            if event.get("Type") == "container" and event.get("Action") == "die":
                container_id = None
                container_name = None
                if "Actor" in event:
                    if "Attributes" in event["Actor"]:
                        container_name = event["Actor"]["Attributes"].get("name")
                    container_id = event["Actor"].get("ID")
                if callback and container_id:
                    callback(container_id, "die", container_name)
                die_events.append(event)
        return die_events

    def get_latest_container_to_pid_mapping(self, pid_callback=None):
        if pid_callback:
            print("Merging container events with PID and metrics updates...")
        return self.cgroups.get_container_names_to_pids()

    def merge_containers_with_pids_from_deltas(self, deltas):
        container_to_pids = self.cgroups.get_container_names_to_pids()
        if not container_to_pids:
            logging.info("No containers or PIDs found.")
            return

        for container_name, pids in container_to_pids.items():
            matching_pids = [pid for pid in pids if pid in deltas]
            missing_pids = [pid for pid in pids if pid not in deltas]

            self.get_container_deltas_summed(container_name, matching_pids, deltas)
            self.get_container_pids_deltas(container_name, matching_pids, deltas)

            logging.info(f"Container: {container_name}")
            logging.info(
                f"  PID count: {len(pids)} | Metrics found for: {len(matching_pids)} | Missing: {len(missing_pids)}"
            )
            if missing_pids:
                logging.debug(f"  Missing PIDs: {missing_pids}")

            # Pretty print per-PID metrics as a table
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
                            "CacheMiss": m.get("cache_misses", 0),
                        }
                    )
                logging.info(
                    "\n" + tabulate(per_pid_metrics, headers="keys", tablefmt="github")
                )

                # Summed metrics
                summed = self.docker_container_to_pids_to_metrics_summed[container_name]
                logging.info("  [SUMMED METRICS]")
                for k, v in summed.items():
                    logging.info(f"    {k:20}: {v}")

                # Validation: sum per-PID and compare to container sum
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
                # logging.info("  [VALIDATION]")
                # for k, (pid_sum, cont_sum, status) in validation.items():
                #     logging.info(
                #         f"    {k:20}: per-PID sum={pid_sum} | container sum={cont_sum} [{status}]"
                #     )
            else:
                logging.info("  No matching PIDs with metrics for this container.")

    def get_container_pids_deltas(self, container, matching_pids, deltas):
        if container not in self.docker_container_to_pids_to_metrics:
            self.docker_container_to_pids_to_metrics[container] = {}
        for pid in matching_pids:
            self.docker_container_to_pids_to_metrics[container][pid] = deltas[pid]
        # Show example metrics for one pid of this container at debug level
        if matching_pids:
            example_pid = random.choice(matching_pids)
            example_metrics = deltas[example_pid]
            logging.debug(
                f"  Example metrics for container '{container}', pid {example_pid}: {example_metrics}"
            )
        return self.docker_container_to_pids_to_metrics

    def get_container_deltas_summed(self, container, matching_pids, deltas):
        exclude_keys = {"pid", "ppid", "name"}
        metrics_list = []
        for pid in matching_pids:
            if pid in deltas:
                filtered = {
                    k: v for k, v in deltas[pid].items() if k not in exclude_keys
                }

                # Flatten syscall class deltas so container-level sums keep these features.
                syscall_class_deltas = filtered.pop("syscall_class_deltas", {}) or {}
                if isinstance(syscall_class_deltas, dict):
                    for cls, count in syscall_class_deltas.items():
                        filtered[f"syscall_class_{cls}"] = count

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
        self.docker_container_to_pids_to_metrics_summed[container] = summed_metrics
        return self.docker_container_to_pids_to_metrics_summed
