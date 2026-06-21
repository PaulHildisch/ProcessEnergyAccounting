from model_builder import ModelBuilder
from preprocessing import Preprocessor
from plotting import Plotter
import pandas as pd


DATA_PATH  = pd.read_parquet("data/process_interval_data_wf.parquet")
data  = pd.read_parquet(DATA_PATH)

good_features = ['context_switches', 'syscall_class_network', 'syscall_class_sched', 'delta_cycles', 'delta_branch_instructions', 'syscall_class_signal']
alg_name = "ebm"

preprocessor = Preprocessor(data, good_features)
X, y, times = preprocessor.preprocess()

builder = ModelBuilder(X, y, times, alg_name)
y_pred, y_test, t_test = builder.run_and_save_model("model_testing/clean_impl_ebm/")
plotter = Plotter(y_pred,y_test, t_test, window_start =50, window_end=200)
plotter.plot_and_save("model_testing/clean_impl_ebm/")

#Build plotter - done
#Load model
#Think about incremental fitting
#Build shapley explainer
