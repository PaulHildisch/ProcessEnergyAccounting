import logging
import queue
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
        perf_events="auto",
        enable_perf_counters=True,
        enable_hardware_metrics=True,
    ):
        self.monitor = MonitoringClient(
            model_features=getattr(online_estimator, "features", None),
            perf_events=perf_events,
            enable_perf_counters=enable_perf_counters,
        )
        self.hw_profiler = HardwareProfiler() if enable_hardware_metrics else None
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
        self.db_write_queue = queue.Queue() if self.db_client else None
        self.db_writer_thread = (
            threading.Thread(target=self._db_writer_loop, daemon=True)
            if self.db_client
            else None
        )
        self.meter_lock = threading.Lock()
        self.meter_sum = 0.0
        self.meter_count = 0
        self.meter_thread = (
            threading.Thread(target=self._meter_loop, daemon=True)
            if self.meter_client
            else None
        )

    def start(self):
        self.running = True
        if self.db_writer_thread is not None:
            self.db_writer_thread.start()
        if self.meter_thread is not None:
            self.meter_thread.start()
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=self.interval + 1.0)
        if self.db_write_queue is not None:
            self.db_write_queue.put(None)
        if self.db_writer_thread is not None:
            self.db_writer_thread.join(timeout=5.0)
        if self.meter_thread is not None:
            self.meter_thread.join(timeout=5.0)
        self.monitor.close()

    def _collect(self):
        interval_start_wall = time.time()
        next_deadline = time.monotonic() + self.interval
        next_deadline_wall = interval_start_wall + self.interval
        while self.running:
            try:
                while self.running:
                    remaining = next_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.1))

                if not self.running:
                    break

                interval_end = next_deadline_wall
                fetch_started = time.monotonic()
                process_data = self.monitor.get_process_list()
                fetch_duration = time.monotonic() - fetch_started
                logging.info("Metric fetch took %.3fs", fetch_duration)
                self.snapshots.append((interval_end, process_data))

                avg_power = self._consume_avg_power()
                actual_interval = interval_end - interval_start_wall
                # avg_power is measured in watts and actual_interval in seconds,
                # so interval_energy is stored in joules (W * s).
                interval_energy = avg_power * actual_interval

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

                    if self.exporter is not None:
                        exporter_mode = getattr(self.exporter, "mode", "process")
                        if exporter_mode == "process":
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
                        elif exporter_mode == "container":
                            logging.info(
                                "Prometheus exporter mode 'container' selected with %s aggregated containers available.",
                                len(container_metrics),
                            )
                            self.exporter.set_container_metrics(
                                timestamp=interval_end,
                                interval=interval,
                                node="localhost",
                                container_metrics=container_metrics,
                            )
                        elif exporter_mode == "pod":
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

                    # Push deltas to aggregation layer without blocking the collector loop.
                    if deltas and self.db_client and self.meter_client is None:
                        print(f"[{time.strftime('%X')}] delta count: {len(deltas)}")
                        if self.db_write_queue is not None:
                            self.db_write_queue.put(
                                {
                                    "timestamp": interval_end,
                                    "interval": interval,
                                    "deltas": deltas,
                                    "pid_to_container": pid_to_container,
                                }
                            )
                    elif self.db_client and self.meter_client:
                        print(
                            f"[{time.strftime('%X')}] delta count: {len(deltas)}, avg_power_w: {avg_power}, interval_energy_j: {interval_energy}"
                        )
                        if self.db_write_queue is not None:
                            self.db_write_queue.put(
                                {
                                    "timestamp": interval_end,
                                    "interval": interval,
                                    "deltas": deltas,
                                    "interval_energy": interval_energy,
                                    "avg_power": avg_power,
                                    "pid_to_container": pid_to_container,
                                }
                            )

                after_work = time.monotonic()
                missed_by = after_work - next_deadline
                if missed_by > 0.001:
                    logging.warning(
                        "Collector missed interval deadline by %.3fs", missed_by
                    )
                    intervals_missed = int(missed_by // self.interval)
                    deadline_step = (intervals_missed + 1) * self.interval
                    next_deadline += deadline_step
                    next_deadline_wall += deadline_step
                else:
                    next_deadline += self.interval
                    next_deadline_wall += self.interval
                interval_start_wall = interval_end

            except Exception:
                logging.exception("Collector loop failed")
                self.running = False

    def _db_writer_loop(self):
        if self.db_write_queue is None or self.db_client is None:
            return
        while True:
            item = self.db_write_queue.get()
            if item is None:
                break
            try:
                self.db_client.write_deltas(**item)
            except Exception:
                logging.exception("DB writer failed")

    def _meter_loop(self):
        while self.running:
            sample_started = time.monotonic()
            power = self._sample_power()
            if power is not None:
                with self.meter_lock:
                    self.meter_sum += power
                    self.meter_count += 1
            sample_elapsed = time.monotonic() - sample_started
            sleep_time = max(self.sample_rate - sample_elapsed, 0.0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _consume_avg_power(self):
        with self.meter_lock:
            if self.meter_count == 0:
                return 0.0
            avg_power = self.meter_sum / self.meter_count
            self.meter_sum = 0.0
            self.meter_count = 0
            return avg_power

    def _sample_power(self):
        if self.meter_client is None:
            return None
        try:
            sensor_ids = set(self.meter_sensor_ids)
            readings = [
                sensor.get("data", {}).get("ActivePower")
                for sensor in self.meter_client.get_sensor_data()
                if sensor.get("id") in sensor_ids or sensor.get("name") in sensor_ids
            ]
            readings = [reading for reading in readings if reading is not None]
            return sum(readings) if readings else None
        except Exception as exc:
            logging.warning("Error fetching power data: %s", exc)
            return None

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

    perf_events = (args.perf_events or "auto").strip().lower()
    hardware_metrics = (args.hardware_metrics or "all").strip().lower()
    enable_perf_counters = perf_events != "no"
    enable_hardware_metrics = hardware_metrics != "no"

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
        perf_events=perf_events,
        enable_perf_counters=enable_perf_counters,
        enable_hardware_metrics=enable_hardware_metrics,
    )

    monitor.start()

    print("Monitoring started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping monitor...")
        try:
            monitor.stop()
        except KeyboardInterrupt:
            print("Forced stop requested; exiting.")
        finally:
            if db_client:
                db_client.close()
            print("Monitoring stopped.")
