# Process Energy Accounting

**Forked from (https://github.com/MPDS-DOS/ProcessEnergyAccounting.git)[https://github.com/MPDS-DOS/ProcessEnergyAccounting.git]**

This repository contains code for collecting process-level runtime metrics, estimating energy usage, and running workloads/experiments for energy accounting research.

## Repository layout

| Path | Purpose |
|---|---|
| [`monitor/`](monitor/) | Runtime monitor for collecting process, Docker container, or Kubernetes pod metrics. Includes Docker and Kubernetes deployment files. |
| [`modeling/`](modeling/) | Model training and pretrained energy-estimation artifacts. |
| [`workload/`](workload/) | Workloads used for experiments and evaluation. |
| [`experiments/`](experiments/) | Experiment scripts/configuration and collected results. |
| [`scripts/`](scripts/) | Helper scripts for setup and automation. |
| [`py-env.sh`](py-env.sh) | Helper script for setting up the Python environment. |

## Setup

Create the Python environment with:

```sh
./py-env.sh
```

Then activate it, for example with fish:

```sh
source .venv/bin/activate.fish
```

or with POSIX shells:

```sh
source .venv/bin/activate
```

## Monitor quick start

The monitor can run as a Python script, Docker container, or Kubernetes DaemonSet.

For detailed monitor usage, see [`monitor/README.md`](monitor/README.md).

Minimal local example:

```sh
sudo .venv/bin/python monitor/delta_aggregator.py \
  --interval 1 \
  --use-prometheus-exporter \
  --exporter-addr localhost \
  --exporter-port 9002 \
  --exporter-mode process
```

## Docker/Kubernetes deployment

Deployment files live in [`monitor/deployment/`](monitor/deployment/).

Typical Kubernetes deployment:

```sh
cd monitor
kubectl apply -f deployment/configmap.yaml
kubectl apply -f deployment/daemonset.yaml
```

## Notes

- The monitor needs privileged host access for `/proc`, `/sys`, cgroups, kernel modules, and perf/BPF-based metrics.
- Model-based online energy estimation is enabled by passing `--model-pkl` to the monitor.
- External power-meter and InfluxDB options are mainly useful for collecting labelled training data.
