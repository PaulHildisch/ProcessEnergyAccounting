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

### Auto vs Generalized vs SFS features

For easier replication we hard coded all paths that were used for the experiments in the report. To run this experiment with one of the datasets uncomment the dataset in the data_map (Running this experiment for all dataset in a row might take up to much memory so we recommend chosing one or two at a time). This pipeline will then compare the performance performance of various models trained on a general feature set with automatic selection and sequential forward selection for the given dataset. Beware this experiment has to be executed like a python module using the -m flag and . as seperators and no .py after the last file. It must be called from the ProcessEnergyAccounting directroy!

```
python3 -m model_testing.clean_impl.experiments.auto_vs_generalized
```

For the deep learning version run:
```
python3 -m model_testing.clean_impl.experiments.auto_vs_generalized_dl
```

### Extended vs Original feature set
This pipeline compares the performance of the original feature set with the extended feature set. Our automatic selection is used to select the features from the possible options. Datasets recorded with the new monitoring/extended feature set are already pre selected. Just run:

```
python3 -m model_testing.clean_impl.experiments.compare_new_vs_original_features
```
Note: Since this experiment used the new monitoring please comment out `self._remove_outliers(window=5, max_deviation_energy= 150)` in the `preprocess_no_split()` function in preprocessing.py, if you want the reproduce the paper results exactly. We did this because the outlier detection seemed too strict for the new monitoring, but remained necessary for the old results.

### Full Prediction and Attribution
This is our end to end prediction and attribution pipeline. This pipeline will train and evaluate a selected model using either general or automatically selected features. The model can be saved for further use. If the model type supports attribution a process level energy attribution can be created. Look at the end of the full_prediction_attribution_pipeline.py file for an example configuration. Calling the `pipeline()` function with "AUTO" will lead to automatic selection, "GENERAL" will use the general feature set. To replicate the attribution from the report choose "AMPLISEQ" as the dataset (standard setting). To run this pipeline use (from the ProcessEnergyAccountig directory):

```
 python3 -m model_testing.clean_impl.pipeline.full_prediction_attribution_pipeline
```

To call the deep learning implementation use:
```
python3 -m model_testing.clean_impl.pipeline.full_prediction_attribution_pipeline_dl
```

