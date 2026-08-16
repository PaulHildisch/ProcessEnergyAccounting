# Process Energy Accounting

**Forked from (https://github.com/MPDS-DOS/ProcessEnergyAccounting.git)[https://github.com/MPDS-DOS/ProcessEnergyAccounting.git]**

This repository contains code for collecting process-level runtime metrics, estimating energy usage, and running workloads/experiments for energy accounting research. As this is a fork, there might be unused artifacts left over by the fork.

## Repository layout
| Path | Purpose |
|---|---|
| [`model_testing/`](model_testing/clean_impl/) | Main pipeline and evaluation code. Also includes specific experiments.|
| [`monitor/`](monitor/) | Runtime monitor for collecting process, Docker container, or Kubernetes pod metrics. Includes Docker and Kubernetes deployment files. |
| [`modeling/`](modeling/) | Model training and pretrained energy-estimation artifacts. |
| [`workload/`](workload/) | Scripts to create workloads used in experiments and evaluation. |
| [`scripts/`](scripts/) | Helper scripts for setup and automation. |
| [`py-env.sh`](py-env.sh) | Helper script for setting up the Python environment. |
| [`.env.example`](.env.example) | Template for the .env file used to configure InfluxDB and Powermeter | 

## Setup
Create and activate the Python environment with:
```sh
./py-env.sh && source .venv/bin/activate
```

To successfully capture kernel level metrics, BCC needs to be compiled and imported into the generated environment. A convenience scripts is provided at [`scripts/install-deps.sh`](scripts/install-deps.sh).

## Data collection quick start
The monitor can run as a Python script, Docker container, or Kubernetes DaemonSet.
For detailed monitor usage, see [`monitor/README.md`](monitor/README.md).

To record data in the format expect by the experiments implementated in [`model_testing/clean_impl/experiments`](model_testing/clean_impl/experiments) use the following monitoring parameters. Configuration for InfluxDB and Powermeter should be provided in a .env file.

```sh
sudo $(which python) monitor/delta_aggregator.py \
  --use-influxdb \
  --use-meter \
  --meter-sensor-id [ID] \
  --perf-events [List of desired features]
```

To generate reproducible stress run `workload/daw/daw-load.sh --pipelines-file workload/daw/nextflow/nfcore_test_pipelines.txt`.

After successfully running the pipelines, the recorded data can be extracted using `modeling/data/db-export.sh --session-dir path/to/nextflow/output/directory/ --output path/to/data/storage/file.parquet`.

## Notes
- The monitor needs privileged host access for `/proc`, `/sys`, cgroups, kernel modules, and perf/BPF-based metrics.
- External power-meter and InfluxDB options are mainly useful for collecting labelled training data.
