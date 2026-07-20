# Process Monitor

Host-level process monitoring with optional Docker/Kubernetes aggregation, Prometheus export, InfluxDB export, smart-meter input, and model-based energy estimation.

The monitor needs host access to `/proc`, `/sys`, cgroups, kernel modules, and usually root/privileged permissions.

## Run as a Python script

From the repository root or from `monitor/`, use the project virtualenv Python when root privileges are needed:

```sh
sudo /path/to/your/.venv/bin/python monitor/delta_aggregator.py \
  --interval 1 \
  --use-prometheus-exporter \
  --exporter-addr localhost \
  --exporter-port 9002 \
  --exporter-mode process
```

With Kubernetes pod aggregation:

```sh
sudo /path/to/your/.venv/bin/python monitor/delta_aggregator.py \
  --interval 1 \
  --kubernetes-integration \
  --kubeconfig /etc/rancher/k3s/k3s.yaml \
  --use-prometheus-exporter \
  --exporter-addr localhost \
  --exporter-port 9002 \
  --exporter-mode pod
```

With model-based process energy estimation:

```sh
sudo /path/to/your/.venv/bin/python monitor/delta_aggregator.py \
  --interval 1 \
  --model-pkl ../modeling/estimation/pretrained-models/<host-name>.pkl \
  --use-prometheus-exporter \
  --exporter-addr localhost \
  --exporter-port 9002 \
  --exporter-mode process
```

## Run with Docker

Build the image from `monitor/`:

```sh
docker build -t energy-monitor:local .
```

Run in process mode:

```sh
sudo docker run --rm -it \
  --privileged --pid=host --network=host \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
  -v /proc:/proc:rw \
  -v /sys:/sys:rw \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  energy-monitor:local \
  python delta_aggregator.py \
    --interval 1 \
    --use-prometheus-exporter \
    --exporter-addr localhost \
    --exporter-port 9002 \
    --exporter-mode process
```

Run with Kubernetes integration by also mounting the kubeconfig:

```sh
sudo docker run --rm -it \
  --privileged --pid=host --network=host \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
  -v /proc:/proc:rw \
  -v /sys:/sys:rw \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /etc/rancher/k3s/k3s.yaml:/etc/rancher/k3s/k3s.yaml:ro \
  energy-monitor:local \
  python delta_aggregator.py \
    --interval 1 \
    --kubernetes-integration \
    --kubeconfig /etc/rancher/k3s/k3s.yaml \
    --exporter-mode pod
```

## Deploy on Kubernetes

The Kubernetes deployment uses:

- `deployment/configmap.yaml` for per-node CLI arguments
- `deployment/daemonset.yaml` to run one monitor pod per node

Apply both:

```sh
kubectl apply -f deployment/configmap.yaml
kubectl apply -f deployment/daemonset.yaml
```

Check it:

```sh
kubectl get pods -l app=energy-monitor -o wide
kubectl logs -f <pod-name>
```

Restart after changing the ConfigMap:

```sh
kubectl rollout restart daemonset energy-monitor
```

## CLI options

### Monitoring and aggregation

Use these flags when you want to observe process/container/pod metrics and export them, for example to Prometheus.

| Option | Default | Description |
|---|---:|---|
| `--interval` | `2.0` | Aggregation window in seconds. |
| `--sample-rate` | same as interval | Sampling rate in seconds. |
| `--docker-integration` | off | Aggregate process metrics by Docker container. |
| `--kubernetes-integration` | off | Aggregate process metrics by Kubernetes pod. |
| `--kubeconfig` | unset | Kubeconfig path. Required with `--kubernetes-integration`. |
| `--use-pod-regex` | off | Filter pod names using `K8S_POD_REGEX` from `monitor/.env`. |
| `--debug` | off | Enable debug logging. |

### Prometheus exporter

Use these flags when the monitor should expose live metrics for scraping.

| Option | Default | Description |
|---|---:|---|
| `--use-prometheus-exporter` | off | Enable Prometheus metrics endpoint. |
| `--exporter-addr` | `EXPORTER_ADDR` env | Prometheus bind address, e.g. `localhost` or `0.0.0.0`. |
| `--exporter-port` | `EXPORTER_PORT` env | Prometheus port, e.g. `9002`. |
| `--exporter-mode` | `process` | Export granularity: `process`, `container`, or `pod`. |

### Training data collection and external power meter

Use these flags when collecting labelled data for model training. The external smart meter provides the measured host power/energy target, and InfluxDB can store the resulting interval data.

| Option | Default | Description |
|---|---:|---|
| `--use-influxdb` | off | Write metrics to InfluxDB. Useful for later training/analysis. |
| `--influx-url` | `INFLUX_URL` env | InfluxDB URL. |
| `--influx-token` | `INFLUX_TOKEN` env | InfluxDB token. |
| `--influx-org` | `INFLUX_ORG` env | InfluxDB organization. |
| `--influx-bucket` | `INFLUX_BUCKET` env | InfluxDB bucket. |
| `--use-meter` | off | Enable external smart-meter reading. |
| `--meter-host` | `SMARTMETER_HOST` env | Smart-meter host. |
| `--meter-user` | `SMARTMETER_USER` env | Smart-meter username. |
| `--meter-password` | `SMARTMETER_PASSWORD` env | Smart-meter password. |
| `--meter-ssl` | off / `SMARTMETER_SSL` env | Use SSL for the smart-meter client. |
| `--meter-sensor-id` | `L1` | Smart-meter sensor id. Multiple ids can be comma-separated. |

### Online energy estimation

Use these flags when applying an already trained model during monitoring.

| Option | Default | Description |
|---|---:|---|
| `--model-pkl` | unset | Load a trained model pickle and enable online energy estimation. |
| `--online-energy-estimation` | off | Legacy placeholder flag; model estimation is enabled by `--model-pkl`. |

## Metric overview

Most metrics are reported as interval deltas, i.e. the change between two monitor snapshots.

### Process and system metrics

| Metric | Description |
|---|---|
| `pid` | Process ID. |
| `ppid` | Parent process ID. |
| `name` | Process name. |
| `delta_cpu_ns` | CPU runtime delta from BPF/proc source, in nanoseconds. |
| `delta_cpu_time_proc` | CPU time delta derived from `/proc/<pid>/stat` ticks. |
| `delta_cpu_time_psutil` | CPU time delta from `psutil`. |
| `delta_rss_memory` | Change in resident memory size in bytes. |
| `context_switches` | Context-switch count delta. |
| `syscall_count` | Total syscall count delta. |
| `syscall_class_<class>` | Syscall count delta grouped by class, e.g. file, network, memory, process, sched, signal, time, or other. |

### I/O and network metrics

| Metric | Description |
|---|---|
| `delta_io_bytes` | Combined disk I/O byte delta. |
| `delta_disk_read_bytes` | Disk read byte delta. |
| `delta_disk_write_bytes` | Disk write byte delta. |
| `delta_net_send_bytes` | Network send byte delta. |
| `delta_net_recv_bytes` | Network receive byte delta. |
| `delta_net_send_packets` | Network send packet delta. |
| `delta_net_recv_packets` | Network receive packet delta. |

### Perf hardware/software counters

Perf counters are collected persistently per PID and then differenced per interval. If a model is loaded, only the perf counters needed by the model are opened to avoid too many file descriptors.

| Metric | Description |
|---|---|
| `delta_instructions` | Retired instruction count delta. |
| `delta_cycles` | CPU cycle count delta. |
| `delta_ref_cpu_cycles` | Reference CPU cycle count delta, less affected by frequency scaling. |
| `delta_branch_instructions` | Retired branch instruction count delta. |
| `delta_branch_misses` | Branch misprediction count delta. |
| `delta_cache_misses` | Generic hardware cache miss count delta. |
| `delta_stalled_cycles_backend` | Backend stalled cycle count delta. |
| `delta_stalled_cycles_frontend` | Frontend stalled cycle count delta. |
| `delta_llc_load_misses` | Last-level cache load miss delta. |
| `delta_llc_store_misses` | Last-level cache store miss delta. |
| `delta_l1d_load_misses` | L1 data cache load miss delta. |
| `delta_dtlb_load_misses` | Data TLB load miss delta. |
| `delta_dtlb_store_misses` | Data TLB store miss delta. |
| `delta_node_load_misses` | NUMA node/remote memory load miss delta, if supported. |
| `delta_cpu_migrations` | CPU migration count delta. |
| `delta_page_faults_min` | Minor page fault count delta. |
| `delta_page_faults_maj` | Major page fault count delta. |

### FP/SIMD perf counters

These are CPU-vendor-specific raw perf counters. Only counters supported by the host are exposed.

| Metric | Description |
|---|---|
| `delta_fp_scalar` | Intel: retired scalar floating-point operations. |
| `delta_fp_128b_packed` | Intel: retired 128-bit packed floating-point operations. |
| `delta_fp_256b_packed` | Intel: retired 256-bit packed floating-point operations. |
| `delta_fp_512b_packed` | Intel: retired 512-bit packed floating-point operations, if AVX-512 is available. |
| `delta_fp_add_sub` | AMD: retired floating-point add/subtract operations. |
| `delta_fp_mult` | AMD: retired floating-point multiply operations. |
| `delta_fp_div` | AMD: retired floating-point divide operations. |
| `delta_fp_mac` | AMD: retired floating-point multiply-accumulate operations. |

### Hardware context features

These describe the host platform/operating point and can be stored with process metrics for cross-platform models.

| Metric | Description |
|---|---|
| `arch` | CPU architecture, e.g. x86_64 or arm64. |
| `cpu_vendor` | CPU vendor, e.g. Intel, AMD, ARM, Apple, or other. |
| `tdp_tier` | Approximate TDP class: low, mid, high, or unknown. |
| `cpu_governor` | Current CPU frequency governor, e.g. performance, powersave, or schedutil. |
| `numa_node_count` | Number of NUMA nodes visible on the host. |
| `freq_ratio` | Current CPU frequency ratio relative to max frequency. |

### Power and prediction fields

| Metric | Description |
|---|---|
| `avg_power` | Average external smart-meter power over the interval, when `--use-meter` is enabled. |
| `interval_energy` | External smart-meter energy estimate for the interval, when `--use-meter` is enabled. |
| `predicted_energy` | Model-predicted energy for a process/container/pod when `--model-pkl` is used. |

## Kubernetes ConfigMap

`deployment/configmap.yaml` contains the CLI arguments used by the DaemonSet.

Example:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: delta-aggregator-args
  namespace: default
data:
  default: >-
    --interval 1
    --kubernetes-integration
    --kubeconfig /etc/rancher/k3s/k3s.yaml

  tux: >-
    --interval 1
    --kubernetes-integration
    --kubeconfig /etc/rancher/k3s/k3s.yaml
    --model-pkl /app/models/siena06.pkl
    --use-prometheus-exporter
    --exporter-port 9002
    --exporter-addr localhost
    --exporter-mode process
```

| Key | Meaning |
|---|---|
| `default` | Fallback arguments for nodes without a matching key. |
| `<node-name>` | Per-node override. Must match `spec.nodeName`, e.g. `tux`. |
| `--interval 1` | Collect one-second metric intervals. |
| `--kubernetes-integration` | Watch Kubernetes pods and map pod names to PIDs. |
| `--kubeconfig ...` | Kubeconfig mounted into the pod by the DaemonSet. |
| `--model-pkl ...` | Model file inside the image, e.g. `/app/models/siena06.pkl`. |
| `--use-prometheus-exporter` | Start Prometheus exporter inside the monitor pod. |
| `--exporter-port 9002` | Prometheus exporter port. With `hostNetwork: true`, this is a host port. |
| `--exporter-addr localhost` | Bind address for the exporter. Use `0.0.0.0` if scraping from outside the host namespace. |
| `--exporter-mode process` | Export process-level metrics. Use `pod` for pod-level aggregation. |

The DaemonSet picks the args file like this:

```sh
ARGS_FILE="/etc/aggregator-args/$K8S_NODE_NAME"
[ -f "$ARGS_FILE" ] || ARGS_FILE="/etc/aggregator-args/default"
exec python delta_aggregator.py $(cat "$ARGS_FILE")
```

So add one ConfigMap entry per node only when that node needs custom arguments.
