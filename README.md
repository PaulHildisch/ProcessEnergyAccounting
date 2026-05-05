# Workflow Energy Estimations

This project learns a regression model that predicts per-process energy usage from process-level runtime metrics plus measured node power. Workflow attribution is a separate, post-hoc step used for aggregation and analysis after the model has been trained.

## Intended Workflow

The operational model is:

1. start the monitor in one `tmux` pane
2. start nf-core load generation in another `tmux` pane
3. let the load generator run several `-profile test` pipelines with random idle periods between them
4. stop the monitor when the load-generation session is complete
5. export a process-level parquet dataset from InfluxDB
6. train the regression model on process rows only
7. if you want workflow/job insights, annotate the process dataset afterward using the saved Nextflow traces
8. aggregate estimated process energy to task/workflow level

That separation is deliberate:

- monitoring is independent of Nextflow
- model training is independent of workflow attribution
- workflow attribution is only required for post-hoc reporting and aggregation

## Repository Layout

- `delta_aggregator.py`: continuous process and power monitoring
- `monitoring/`: BPF counters, `/proc`/`psutil` metadata, and Nextflow trace resolver
- `database/client.py`: InfluxDB read/write logic
- `smart_meter/client.py`: smart meter API client
- `scripts/run_nfcore_load_generation.sh`: nf-core test-profile load generator with random idle periods
- `scripts/export_process_dataset.sh`: export a monitored session from InfluxDB to process-level parquet
- `nextflow/trace.config`: Nextflow trace field configuration
- `nextflow/nfcore_test_pipelines.txt`: starter list of nf-core pipelines for load generation
- `estimation/data/data_loader.py`: export process/task/workflow datasets from InfluxDB
- `estimation/train_energy_regression.py`: process-only regression training CLI
- `estimation/annotate_nextflow_runs.py`: post-hoc workflow attribution for process datasets
- `estimation/estimate_energy.py`: apply a trained model to process data and optionally aggregate estimates
- `estimation/analyze_energy.py`: higher-level workflow/task analysis

## Requirements

- Python packages:
    - `pandas`
    - `numpy`
    - `psutil`
    - `influxdb-client`
    - `requests`
    - `cvxpy`
    - `scikit-learn`
    - `matplotlib`
    - `pyarrow` or `fastparquet`
- BCC/BPF support for the kernel and userspace tools
- InfluxDB 2.x
- `nextflow`
- a working Nextflow execution backend for nf-core test profiles
- a smart meter reachable through [`smart_meter/client.py`](/Users/juliusirion/repos/j-irion-github/workflow-energy-estimations/smart_meter/client.py)

## Node Setup

On cluster or lab nodes, the monitor usually needs root privileges because it loads BCC tracepoint probes.

The validated setup on a node like `siena06` is:

1. use the system Python for the Poetry environment, not a newer standalone Python
2. allow the Poetry environment to see system site packages so the OS-installed `bcc` bindings are available
3. run the monitor with `sudo`

Example:

```bash
poetry config virtualenvs.options.system-site-packages true --local
poetry env remove --all
poetry env use /usr/bin/python3
poetry lock
poetry install
```

Then verify:

```bash
poetry run python --version
poetry run python -c "from bcc import BPF; import psutil; from dotenv import load_dotenv; print('ok')"
```

On nodes where BCC tracepoints require elevated privileges, start the monitor like this:

```bash
sudo .venv/bin/python delta_aggregator.py --interval 2 --sample-rate 0.5 --meter-ssl
```

## InfluxDB

The repository includes a minimal local InfluxDB stack in [`docker-compose.yml`](/Users/juliusirion/repos/j-irion-github/workflow-energy-estimations/docker-compose.yml).

```bash
docker compose up -d
```

## Configuration

The monitor and export CLI load a local `.env` file automatically.

Relevant variables:

- `INFLUX_URL`
- `INFLUX_TOKEN`
- `INFLUX_ORG`
- `INFLUX_BUCKET`
- `SMARTMETER_HOST`
- `SMARTMETER_USER`
- `SMARTMETER_PASSWORD`
- `SMARTMETER_SSL`

Example:

```dotenv
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=my-super-secret-auth-token
INFLUX_ORG=myorg
INFLUX_BUCKET=mybucket
SMARTMETER_HOST=<meter-host>
SMARTMETER_USER=<meter-user>
SMARTMETER_PASSWORD=<meter-password>
SMARTMETER_SSL=false
```

Note:

- set `SMARTMETER_SSL=true` if your meter only works over HTTPS
- you can still override that with `--meter-ssl` on the CLI

## Step 1: Start Monitoring

In the first `tmux` pane, start the monitor. It will collect power and process metrics for all live PIDs continuously, regardless of whether a Nextflow pipeline is running.

```bash
sudo .venv/bin/python delta_aggregator.py \
  --interval 2 \
  --sample-rate 0.5 \
  --meter-ssl
```

You can still pass explicit `--influx-*` and `--meter-*` arguments if you do not want to rely on `.env`.

Important:

- do not pass `--nextflow-trace` for normal training-data collection
- the monitor should stay generic and observe the whole machine
- idle periods are simply periods where no nf-core pipeline is active, but normal system processes still exist
- on restricted nodes, root privileges may be required for BCC tracepoint loading

## Step 2: Run nf-core Load Generation

In the second `tmux` pane, run the load-generation script. It enforces `-profile test` for every pipeline and appends a backend profile (default: `docker`), then inserts random idle periods between runs.

The default pipeline list is [`nextflow/nfcore_test_pipelines.txt`](/Users/juliusirion/repos/j-irion-github/workflow-energy-estimations/nextflow/nfcore_test_pipelines.txt).

Example:

```bash
bash scripts/run_nfcore_load_generation.sh \
  --pipelines-file nextflow/nfcore_test_pipelines.txt \
  --backend-profile docker \
  --idle-min 30 \
  --idle-max 180 \
  --initial-idle 60 \
  --final-idle 60
```

You can also specify pipelines directly:

```bash
bash scripts/run_nfcore_load_generation.sh \
  --backend-profile docker \
  --pipeline "nf-core/scnanoseq -r 1.2.2" \
  --pipeline "nf-core/rnaseq"
```

Each pipeline is run like this:

```bash
nextflow run nf-core/scnanoseq -r 1.2.2 -profile test,docker --outdir <OUTDIR>
```

The script additionally enables:

- `-c nextflow/trace.config`
- `-with-trace`
- `-with-report`
- `-with-timeline`

Output goes to `runs/<session_id>/` and includes:

- `manifest.tsv`
- `session_start.txt`
- `session_stop.txt`
- per-pipeline `trace.txt`, `report.html`, and `timeline.html`
- combined `nextflow/session_trace.tsv`

## Step 3: Stop Monitoring

When the load-generation script finishes, stop `delta_aggregator.py` in the first pane.

At that point you have:

- raw monitored process/power data in InfluxDB
- Nextflow traces and run metadata on disk under `runs/<session_id>/`

## Step 4: Export the Process-Level Training Dataset

Export the monitored time range from InfluxDB. Use the timestamps from the load-generation session, or the exact monitor start/stop times if you want a slightly wider range.

Preferred helper:

```bash
bash scripts/export_process_dataset.sh \
  --session-dir runs/<session_id>
```

Equivalent direct export:

```bash
python estimation/data/data_loader.py \
  --level process \
  --start 2026-03-12T10:00:00Z \
  --stop 2026-03-12T11:00:00Z \
  --aggregate-every 1s \
  --output runs/<session_id>/datasets/process_interval_data.parquet
```

The exported process dataset includes the process identity fields needed for post-hoc attribution later, including `pid`, `ppid`, `cmdline`, `exe`, `cwd`, `cgroup`, `create_time`, and `session_id`.

## Step 5: Train the Model

Train on process rows only:

```bash
python estimation/train_energy_regression.py \
  --data runs/<session_id>/datasets/process_interval_data.parquet \
  --model-output artifacts/models/process_energy_regression.pkl
```

This model is intentionally workflow-agnostic. It learns from:

- per-process metrics
- measured interval energy

It does not need workflow labels at training time.

## Step 6: Apply the Trained Model to Other Data

For a new monitored session, export another process dataset and estimate per-process energy:

```bash
python estimation/estimate_energy.py \
  --model artifacts/models/process_energy_regression.pkl \
  --data runs/<other_session_id>/datasets/process_interval_data.parquet \
  --level process \
  --output artifacts/estimates/process_estimates.parquet
```

This gives you process-level energy estimates without requiring workflow attribution.

## Step 7: Add Workflow Attribution Afterward

If you want workflow-, pipeline-, or task-level insights, annotate the exported process dataset using the saved Nextflow traces from the load-generation session:

```bash
python estimation/annotate_nextflow_runs.py \
  --data runs/<session_id>/datasets/process_interval_data.parquet \
  --session-dir runs/<session_id> \
  --output runs/<session_id>/datasets/process_interval_data_attributed.parquet
```

This step uses the saved trace data to add:

- `workflow_run_id`
- `pipeline_name`
- `task_id`
- `task_name`
- `task_tag`
- `work_dir`
- `native_id`
- `group_id`

Training still does not depend on this step. Aggregation does.

## Step 8: Aggregate Estimated Process Energy

Once you have an attributed process dataset, you can aggregate estimated process energy to task or workflow level.

Task-level estimates:

```bash
python estimation/estimate_energy.py \
  --model artifacts/models/process_energy_regression.pkl \
  --data runs/<session_id>/datasets/process_interval_data_attributed.parquet \
  --level task \
  --output artifacts/estimates/task_estimates.parquet
```

Workflow-level estimates:

```bash
python estimation/estimate_energy.py \
  --model artifacts/models/process_energy_regression.pkl \
  --data runs/<session_id>/datasets/process_interval_data_attributed.parquet \
  --level workflow \
  --output artifacts/estimates/workflow_estimates.parquet
```

## Analysis

For attributed datasets, use the higher-level analysis CLI:

```bash
python estimation/analyze_energy.py \
  --data runs/<session_id>/datasets/process_interval_data_attributed.parquet \
  --workflow-summary-output artifacts/reports/workflow_energy_summary.parquet \
  --task-summary-output artifacts/reports/task_energy_summary.parquet \
  --top-n 10
```

These answer questions such as:

- energy per workflow run
- energy per pipeline process/task
- top tasks by energy
- task-level energy contribution over time

## Notes

- `-profile test` is enforced in the load-generation script because full nf-core runs are too slow for iterative dataset collection.
- the script appends a backend profile (default `docker`) so tools like `fastqc` run in the intended execution environment.
- Background operating-system activity during idle periods remains part of the measured process dataset. That is useful, because the model should learn dynamic energy on top of real machine baseline behavior.
- The current post-hoc resolver prefers `native_id` and `work_dir`, constrains matches by the task runtime window from the Nextflow trace, then propagates assignments through the process tree via `ppid`.

## Troubleshooting

- `ModuleNotFoundError: No module named 'bcc'`:
  use the system Python in the Poetry environment and enable system site packages as shown in `Node Setup`.
- `Need super-user privileges to run` or `Operation not permitted` from BCC:
  start the monitor with `sudo`.
- `Connection refused` when talking to the smart meter:
  verify the host and use `--meter-ssl` if the device only serves HTTPS.
- `InsecureRequestWarning` from `urllib3`:
  this is expected when talking to an HTTPS smart meter without certificate verification.
- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`:
  this can happen when the Poetry environment sees older system packages such as `numexpr` or `bottleneck` through `system-site-packages`. The monitor still ran successfully in this state on the validated node, but the cleaner long-term fix is to pin `numpy<2` or install compatible versions of those optional packages in the environment.
- `cannot schedule new futures after shutdown` on Ctrl-C:
  this can appear during InfluxDB client shutdown after the monitor stops. It is a shutdown-time cleanup issue, not a collection-time failure.
