# Model Testing Experiments and Implementation



This folder contains our experiments and pipeline code for data preprocessing, model training, model evaluation and energy attribution. Additionally there is some utility tooling.

## Folder layout

| Path | Purpose |
|---|---|
| [`experiments/`](experiments/) | Experimental pipelines to test the performance of different models given different feature selection approaches, both for classical and deep learning models. Also contains an experiment comparing original vs extended feature set.|
| [`pipeline/`](pipeline/) | Contains the main prediction and an attribution pipeline. This will train (and save if selected) a model on the selected data and create a per process energy estimation. |
| [`plotting/`](plotting/) |Helper classes to plot results. |
| [`utils/`](utils/) | Helper scripts to inpsect data or select a time frame from a parquet file. |


## Replicate experiments

### Data
All collected data is present in the `/runs` directory on the Siena12 node. To avoid confusion, the data that was actually used for our project evaluations is grouped into datasets in the according experiment and pipeline files.

### Auto vs Generalized vs SFS features

For easier replication we hard coded all paths that were used for the experiments in the report. To run this experiment with one of the datasets uncomment the dataset in the data_map (Running this experiment for all dataset in a row might take up to much memory so we recommend chosing one or two at a time). This pipeline will then compare the performance performance of various models trained on a general feature set with automatic selection and sequential forward selection for the given dataset. Beware this experiment has to be executed like a python module using the -m flag and . as seperators and no .py after the last file. It must be called from the ProcessEnergyAccounting directroy!

```shell
python3 -m model_testing.clean_impl.experiments.auto_vs_generalized
```

For the deep learning version run:
```shell
python3 -m model_testing.clean_impl.experiments.auto_vs_generalized_dl
```

### Extended vs Original feature set
This pipeline compares the performance of the original feature set with the extended feature set. Our automatic selection is used to select the features from the possible options. Datasets recorded with the new monitoring/extended feature set are already pre selected. Just run:

```shell
python3 -m model_testing.clean_impl.experiments.compare_new_vs_original_features
```
Note: Since this experiment used the new monitoring please comment out `self._remove_outliers(window=5, max_deviation_energy= 150)` in the `preprocess_no_split()` function in preprocessing.py, if you want the reproduce the paper results exactly. We did this because the outlier detection seemed too strict for the new monitoring, but remained necessary for the old results.

### Full Prediction and Attribution
This is our end to end prediction and attribution pipeline. This pipeline will train and evaluate a selected model using either general or automatically selected features. The model can be saved for further use. If the model type supports attribution a process level energy attribution can be created. Look at the end of the full_prediction_attribution_pipeline.py file for an example configuration. Calling the `pipeline()` function with "AUTO" will lead to automatic selection, "GENERAL" will use the general feature set. To replicate the attribution from the report choose "AMPLISEQ" as the dataset (standard setting). To run this pipeline use (from the ProcessEnergyAccountig directory):

```shell
 python3 -m model_testing.clean_impl.pipeline.full_prediction_attribution_pipeline
```

To call the deep learning implementation use:
```shell
python3 -m model_testing.clean_impl.pipeline.full_prediction_attribution_pipeline_dl
```

### Alternative Summed Prediciton Approach
Running the summed prediction approach requires a little different setup as the model and data files have to handled manually.
First train a model on a workflowtype. By default the model gets saved to `full_pipeline_model.joblib`.
```shell
python3 -m model_testing.clean_impl.pipeline.full_prediction_attribution_pipeline
```

Next run the following command to prepare the dataset used as `test_workflows` in the pipeline training for the experiment. You will have to parse the features selected by the pipeline manually. This will use AMPLISEQ with the features used in the exeperiment as an example:
```shell
python3 -m model_testing.clean_impl.experiments.pid-split-attribution.clean_dataset -f runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet --pid-split --features 'delta_cpu_ns' 'delta_io_bytes' 'delta_net_send_bytes' 'context_switches' 'syscall_count' 'syscall_class_network' 'syscall_class_memory' 'syscall_class_sched'
```

Finally the resulting targets and data files can be used to run the experiment. Again using AMPLISEQ as an example:
```shell
python3 -m model_testing.clean_impl.experiments.pid-split-attribution.pid-split-prediction --modelFile full_pipeline_model.joblib --pidDataSource runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707-preprocessed-pid.parquet --targetDataSource runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707-preprocessed-targets.parquet
```

The evaluation results will be printed and a generated plot saved to [`model_testing/clean_impl/experiments/pid-split-attribution/plots`](model_testing/clean_impl/experiments/pid-split-attribution/plots).
