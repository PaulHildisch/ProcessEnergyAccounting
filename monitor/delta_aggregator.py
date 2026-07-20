import logging
import threading
import time
from collections import deque
from pathlib import Path

from cli import parse_args
from config import apply_env_defaults, load_env_values
from config.runtime import (
    create_cgroups_manager,
    create_db_client,
    create_docker_manager,
    create_exporter,
    create_k8s_manager,
    create_meter_client,
    create_online_estimator,
)
from metrics.delta import compute_delta
from monitoring.hardware_profile import HardwareProfiler
from monitoring.monitor_client import MonitoringClient
from monitoring.reporting import log_process_metrics_table

ENV_PATH = Path(__file__).resolve().parent / ".env"


class DeltaAggregator:
    def __init__(
        self,
        interval=1.0,
        sample_rate=0.1,
        db_client=None,
        exporter=None,
        docker_manager=None,
        cgroups_manager=None,
        online_estimator=None,
        k8s_manager=None,
        meter_client=None,
        meter_sensor_id="L1",
    ):
        self.monitor = MonitoringClient(
            model_features=getattr(online_estimator, "features", None)
        )
        self.hw_profiler = HardwareProfiler()
        self.interval = interval
        self.exporter = exporter
        self.docker_manager = docker_manager
        self.cgroups_manager = cgroups_manager
        self.online_estimator = online_estimator
        self.k8s_manager = k8s_manager
        self.sample_rate = sample_rate
        self.db_client = db_client
        self.meter_client = meter_client
        self.meter_sensor_ids = [s.strip() for s in meter_sensor_id.split(",")]
        self.snapshots = deque(maxlen=2)  # Store only last two process metric snapshots
        self.running = False
        self.thread = threading.Thread(target=self._collect, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.monitor.close()

    def _collect(self):
        while self.running:
            try:
                interval_start = time.time()
                avg_power = 0.0
                interval_energy = 0.0
                while (time.time() - interval_start) < self.interval:
                    sample_time = time.time()
                    sleep_time = self.sample_rate - (time.time() - sample_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                interval_end = time.time()
                if self.meter_client:
                    power_samples = []
                    try:
                        power = sum(
                            self.meter_client.get_power_usage(sid)
                            for sid in self.meter_sensor_ids
                        )
                        if power is not None:
                            power_samples.append(power)
                    except Exception as e:
                        print(f"Error fetching power data: {e}")
                    avg_power = (
                        sum(power_samples) / len(power_samples)
                        if power_samples
                        else 0.0
                    )
                    actual_interval = interval_end - interval_start
                    interval_energy = avg_power * actual_interval

                process_data = self.monitor.get_process_list()
                self.snapshots.append((interval_end, process_data))

                if len(self.snapshots) == 2:
                    interval, deltas = self.get_delta()
                    # Send deltas to DockerManager if present
                    if deltas and self.docker_manager is not None:
                        self.docker_manager.merge_containers_with_pids_from_deltas(
                            deltas
                        )
                    pod_metrics = {}
                    if deltas and self.k8s_manager is not None:
                        pod_metrics = (
                            self.k8s_manager.merge_pod_containers_with_pids_from_deltas(
                                deltas
                            )
                        )

                    container_metrics = {}
                    if self.docker_manager is not None:
                        container_metrics = self.docker_manager.docker_container_to_pids_to_metrics_summed

                    # Build a reverse map of pid → container identity so that
                    # write_deltas can tag each InfluxDB point with its runtime
                    # context (docker/k8s/bare). Constructed from the live
                    # CgroupV2 snapshot held by each manager.
                    pid_to_container: dict = {}
                    if self.docker_manager is not None:
                        for (
                            _name,
                            _pids,
                        ) in self.docker_manager.cgroups.get_container_names_to_pids().items():
                            for _pid in _pids:
                                pid_to_container[_pid] = {
                                    "container_runtime": "docker",
                                    "container_name": _name,
                                    "pod_name": "",
                                }
                    if self.k8s_manager is not None:
                        for (
                            _pod_name,
                            _pids,
                        ) in self.k8s_manager.cgroups.get_container_names_to_pids().items():
                            for _pid in _pids:
                                pid_to_container[_pid] = {
                                    "container_runtime": "k8s",
                                    "container_name": "",
                                    "pod_name": _pod_name,
                                }

                    process_energy_predictions = None
                    if self.online_estimator is not None:
                        process_energy_predictions = (
                            self.online_estimator.run_online_estimation(
                                timestamp=interval_end,
                                interval=interval,
                                deltas=deltas,
                                container_metrics=container_metrics,
                                pod_metrics=pod_metrics,
                                mode=self._get_estimation_mode(),
                                exporter=self.exporter,
                                node="localhost",
                            )
                        )

                    if (
                        self.exporter is not None
                        and getattr(self.exporter, "mode", "process") == "process"
                    ):
                        log_process_metrics_table(
                            deltas,
                            process_energy_predictions,
                            model_features=getattr(
                                self.online_estimator, "features", None
                            ),
                        )
                        self.exporter.set_process_metrics(
                            timestamp=interval_end,
                            interval=interval,
                            deltas=deltas,
                            node="localhost",
                        )

                    # Push deltas to aggregation layer
                    if deltas and self.db_client and self.meter_client is None:
                        print(f"[{time.strftime('%X')}] delta count: {len(deltas)}")
                        self.db_client.write_deltas(
                            timestamp=interval_end,
                            interval=interval,
                            deltas=deltas,
                            pid_to_container=pid_to_container,
                        )
                    elif self.db_client and self.meter_client:
                        process_delta_samples = [
                            metrics for _pid, metrics in sorted(deltas.items())[:2]
                        ]
                        logging.info(
                            "delta count=%s avg_power=%s interval_energy=%s sample_process_deltas=%s",
                            len(deltas),
                            avg_power,
                            interval_energy,
                            process_delta_samples,
                        )
                        self.db_client.write_deltas(
                            timestamp=interval_end,
                            interval=interval,
                            deltas=deltas,
                            interval_energy=interval_energy,
                            avg_power=avg_power,
                            pid_to_container=pid_to_container,
                        )
                    elif self.db_client is None and self.meter_client is None:
                        if self.exporter is not None:
                            if self.exporter.mode == "container":
                                logging.info(
                                    "Prometheus exporter mode 'container' selected with %s aggregated containers available.",
                                    len(container_metrics),
                                )
                            elif self.exporter.mode == "pod":
                                logging.info(
                                    "Prometheus exporter mode 'pod' selected with %s aggregated pods available.",
                                    len(pod_metrics),
                                )
                                self.exporter.set_pod_metrics(
                                    timestamp=interval_end,
                                    interval=interval,
                                    node="localhost",
                                    pod_metrics=pod_metrics,
                                )
            except Exception:
                logging.exception("Collector loop failed")
                self.running = False

    def _get_estimation_mode(self):
        if self.exporter is not None:
            return getattr(self.exporter, "mode", "process")
        if self.k8s_manager is not None:
            return "pod"
        if self.docker_manager is not None:
            return "container"
        return "process"

    def get_delta(self):
        return compute_delta(self.snapshots, self.hw_profiler)


if __name__ == "__main__":
    # Load monitor/.env and keep explicit parsed values for robust fallback.
    env_values = load_env_values(ENV_PATH)

    args = parse_args()
    apply_env_defaults(args, env_values)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    sample_rate = args.sample_rate if args.sample_rate is not None else args.interval

    print(
        f"Starting Interval Metric Aggregation: interval={args.interval}, sample_rate={sample_rate}"
    )
    db_client = create_db_client(args, ENV_PATH)
    meter_client, meter_sensor_id = create_meter_client(args, ENV_PATH)
    cgroups_manager = create_cgroups_manager()
    exporter = create_exporter(args)
    docker_manager = create_docker_manager(args, cgroups_manager)
    online_estimator = create_online_estimator(args)
    k8s_manager = create_k8s_manager(args, cgroups_manager)

    monitor = DeltaAggregator(
        interval=args.interval,
        sample_rate=sample_rate,
        db_client=db_client,
        meter_client=meter_client,
        meter_sensor_id=meter_sensor_id,
        exporter=exporter,
        cgroups_manager=cgroups_manager,
        docker_manager=docker_manager,
        online_estimator=online_estimator,
        k8s_manager=k8s_manager,
    )

    monitor.start()

    print("Monitoring started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        monitor.stop()
        if db_client:
            db_client.close()
        print("Monitoring stopped.")
