from time import perf_counter
start_time = perf_counter()

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPRegressor

from keras import layers, optimizers, callbacks, Sequential
import os,warnings

# Custom imports
from model_testing.clean_impl.pipeline.universal_filtering import CustomSpearmanFilter
from model_testing.clean_impl.pipeline.wrappers_dl import SafeMLPWrapper, SafeKerasWrapper
from sklearn.exceptions import ConvergenceWarning
from model_testing.clean_impl.pipeline.model_builder_keras import KerasModelBuilder, ModelBuilderMLP

from model_testing.clean_impl.pipeline.preprocessing import Preprocessor
from model_testing.clean_impl.plotting.plotting import Plotter
from model_testing.clean_impl.plotting.plotting import plot_dataset

# Suppress normal Tensorflow warnings (or Sklearn MLP Convergence Warning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore",category=ConvergenceWarning)


#This is an experimental pipeline that compares model performance of general features vs automatic features vs sfs features
#This will use A LOT of RAM because all data sets are loaded at the same time
#other_path = "../../ProcessEnergyAccounting/"
other_path=""
log_file_path = "eval_results.txt"

train_ampliseq = [
        "runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet",
        "runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet",
        "runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet"

]
test_ampliseq = "runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet"

train_sarek = [
    "runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet",
    "runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet"

]
test_sarek = "runs/nfcore-20260708T212252Z/datasets/sarek3_0907.parquet"

#Be careful what you uncomment -> test data must not be in test data
train_mixed_unseen_type2 = [
    "runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet",
    "runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet",
    "runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"

]
test_mixed_unseen_type2 = "runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"

train_mixed_seen_type = [
    "runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet",
    "runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet",
    "runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"
]
test_mixed_seen_type = "runs/nfcore-20260708T212252Z/datasets/sarek3_0907.parquet"


train_sarek_cpu06 = [
    "runs/cpu06_scondtry/sarek-1786625685.parquet",
    "runs/cpu06_scondtry/sarek-1786655910.parquet"
]
test_sarek_cpu06 = "runs/cpu06_scondtry/sarek-1786686555.parquet"


train_ampliseq_cpu06 = [
    "runs/cpu06_scondtry/ampliseq-1786566355.parquet",
    "runs/cpu06_scondtry/ampliseq-1786576997.parquet"
]
test_ampliseq_cpu06 = "runs/cpu06_scondtry/ampliseq-1786699827.parquet"




#add data sets you want to test
#It is preferably to not run everthing at the same time -> huge memory foot print
data_map = {
    #"ampliseq": (train_ampliseq,test_ampliseq),
    "sarek" : (train_sarek, test_sarek)#,
    #"train_mixed_unseen_type2": (train_mixed_unseen_type2,test_mixed_unseen_type2),
    #"train_mixed_seen_type": (train_mixed_seen_type, test_mixed_seen_type)
    #"ampliseq_cpu06" : (train_ampliseq_cpu06, test_ampliseq_cpu06),
    #"sarek_cpu06" : (train_sarek_cpu06, test_sarek_cpu06)
}

#Total amount of considered features
features = [
    "delta_cpu_ns",
    "delta_io_bytes",
    "delta_net_send_bytes",
    "context_switches",
    "syscall_count",
    "delta_rss_memory",
    "delta_cpu_time_psutil",
    "delta_cpu_time_proc",
    "syscall_class_file",
    "syscall_class_network",
    "syscall_class_memory",
    "syscall_class_process",
    "syscall_class_other",
    "syscall_class_sched",
    "syscall_class_signal",
    "syscall_class_time",
    "delta_cycles",
    "delta_cache_misses",
    "delta_instructions",
    "delta_branch_instructions",
]


generalized_features =  ['delta_io_bytes', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']


#Choose other Keras model if required.
def dynamic_model(model_name, num_features, window_size=20):
    # Dynamic window size for CNN
    cnn_model = Sequential([

        layers.Input(shape=(num_features, window_size)), # (num_features, window_size)
        layers.Conv1D(32, kernel_size=5, padding='same', activation="relu"),
        layers.BatchNormalization(),

        layers.Conv1D(32, kernel_size=5, padding='same', activation="relu"),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    
    ])

    if model_name.lower() == "cnn":
        model = cnn_model
    else:
        ValueError("Model not implemented")
    return model

mlp_model = MLPRegressor(hidden_layer_sizes=(128,32,16),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.0001,
                    max_iter=500,
                    batch_size=64,
                    early_stopping=True,    # Crucial for time-series stability
                    random_state=42)

for name ,value in data_map.items():

    training_data_path = value[0]
    test_data_path = value[1]

    training_data_list = [pd.read_parquet(other_path+path) for path in training_data_path]
    training_data = pd.concat(training_data_list, ignore_index=True)
    test_data = pd.read_parquet(other_path+test_data_path)


    PNG_NAME = name
    print("/n")
    print("Evaluating : ", name)
    # Write into the log file.
    with open(log_file_path, "a") as f:
        f.write("Evaluating : "+ name + "\n")

    #General train
    preprocessor_train = Preprocessor(training_data, generalized_features)
    X_train, y_train, t_train, _ = preprocessor_train.preprocess_no_split()
    #plot_dataset(t_train, y_train, "multi_training_gen_" +name)

    #General test
    preprocessor_test = Preprocessor(test_data, generalized_features)
    X_test, y_test, t_test , X_test_unaggregated = preprocessor_test.preprocess_no_split()
    #plot_dataset(t_test, y_test, "multi_testing_gen_"+name)

    #Auto train
    preprocessor_train_auto = Preprocessor(training_data, features)
    X_train_auto_FULL, y_train_auto, t_train_auto, _ = preprocessor_train_auto.preprocess_no_split()
    #plot_dataset(t_train_auto, y_train_auto, "multi_training_auto_" +name) 


    #Add models that should be compared (Add the models outside the loop)
    models = ["mlp","cnn"]

    #idle_power_is actually idle interval energy
    for model_name in models:
        model_name = model_name.lower()
        #Evaluate general features
        print("Evaluating gen : " + model_name)
        print(generalized_features)

        # Write into the log file.
        with open(log_file_path, "a") as f:
            f.write("Evaluating gen : " + model_name + "\n")
            for feature in generalized_features:
                f.write(f"{feature},")
            f.write("\n")

        # choose window size > 1 for context based model such as CNN (no effect on MLP, only on Keras Mmdels)
        window_size =  20 
        num_features = len(generalized_features)
        training_model = None
        train_start_time = perf_counter()
        if model_name == "mlp":
            training_model = mlp_model
            builder = ModelBuilderMLP(X_train, X_test, y_train, y_test, training_model, StandardScaler(),log_file_path=log_file_path)
        else:
            training_model = dynamic_model(model_name,num_features,window_size) 
            builder = KerasModelBuilder(X_train, X_test, y_train, y_test, training_model, StandardScaler(), 
                window_size=window_size, train_epochs=30,log_file_path=log_file_path)
                
        y_pred, learned_idle_power = builder.run_and_save_model(".", save = False)   
        train_end_time = perf_counter()

        #Plot general feature prediction results
        if model_name == "mlp" or window_size==1:
            plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
        else:
            plotter = Plotter(y_pred=y_pred,y_test=y_test[window_size - 1:], t_test= t_test[window_size - 1:])
        #plotter.plot_and_save("auto_gen_plots/", "pred_gen_" + PNG_NAME +'_' + model_name)
        plotter.plot_and_save("", "pred_gen_" + PNG_NAME +'_' + model_name)

        training_execution_time = train_end_time - train_start_time
        print(f"Model training execution time: {training_execution_time:.2f} seconds\n")
        with open(log_file_path, "a") as f:
            f.write(f"Model training execution time: {training_execution_time:.2f} seconds\n")
            f.write(f"-------\n")

        #Evaluate auto features
        num_features=len(features) #all features
        small_pipiline_used=False
        fs_model = None
        if model_name == "mlp":
            fs_model = SafeMLPWrapper(batch_size=256,n_repeats=10)
        else:
            fs_model = SafeKerasWrapper(dynamic_model(model_name,num_features,1))
            small_pipiline_used=True
        afs_start_time = perf_counter()
        print("Evaluating auto : " + model_name)     
        automatic_feature_selection = Pipeline(steps=[
                ('variance', VarianceThreshold(threshold=0.01)),
                ('decorrelate', CustomSpearmanFilter(threshold=0.80)),
                ('scaler', StandardScaler()),
                ('select_features', SelectFromModel(fs_model, threshold='0.5*median'))
            ])
        automatic_feature_selection_small = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('select_features', SelectFromModel(fs_model, threshold='0.5*median'))
            ])
        
        if small_pipiline_used:
            automatic_feature_selection=automatic_feature_selection_small

        automatic_feature_selection.set_output(transform="pandas")
        automatic_feature_selection.fit_transform(X_train_auto_FULL, y_train_auto)
        good_features = automatic_feature_selection.get_feature_names_out().tolist()
        print("Selected auto columns:")
        print(good_features)
        afs_end_time = perf_counter()

        with open(log_file_path, "a") as f:
            f.write("Evaluating auto : " + model_name + "\n")
            for feature in good_features:
                f.write(f"{feature},")
            f.write("\n")
            
        X_train_auto = X_train_auto_FULL[good_features]
        

        #Auto test set | Has to be recalcualted every round because results depend on the selected model
        preprocessor_test_auto = Preprocessor(test_data, good_features)
        X_test_auto, y_test_auto, t_test_auto , X_test_unaggregated_auto = preprocessor_test_auto.preprocess_no_split()
        #plot_dataset(t_test_auto, y_test_auto, "multi_testing_auto_"+name)
        
        window_size = 20 # choose window size > 1 for context based model such as CNN
        train_start_time = perf_counter()
        if model_name == "mlp":
            training_model = mlp_model
            builder_auto = ModelBuilderMLP(X_train_auto, X_test_auto, y_train_auto, y_test_auto, training_model, StandardScaler(),log_file_path=log_file_path)
        else:
            num_features = len(good_features)
            training_model = dynamic_model(model_name,num_features,window_size) 
            builder_auto = KerasModelBuilder(X_train_auto, X_test_auto, y_train_auto, y_test_auto, training_model, StandardScaler(), 
                window_size=window_size, train_epochs=30,log_file_path=log_file_path)    
                
        y_pred_auto, learned_idle_power_auto = builder_auto.run_and_save_model(".", save=False)
        train_end_time = perf_counter()
        #Plot automatic feature prediction
        if model_name == "mlp" or window_size==1:
            plotter = Plotter(y_pred_auto, y_test_auto, t_test_auto)#, window_start =50, window_end=200)
        else:
            plotter = Plotter(y_pred_auto, y_test_auto[window_size - 1:], t_test_auto[window_size - 1:])

        #plotter.plot_and_save("auto_gen_plots/", "pred_auto_" + PNG_NAME +'_' + model_name)
        plotter.plot_and_save("", "pred_auto_" + PNG_NAME +'_' + model_name)

        afs_execution_time = afs_end_time - afs_start_time
        training_execution_time = train_end_time - train_start_time
        print(f"AFS execution time: {afs_execution_time:.2f} seconds\n")
        print(f"Model training execution time: {training_execution_time:.2f} seconds\n")
        with open(log_file_path, "a") as f:
            f.write(f"AFS execution time: {afs_execution_time:.2f} seconds\n")
            f.write(f"Model training execution time: {training_execution_time:.2f} seconds\n")
            f.write(f"-------\n")        

        #uncomment for sfs

        if model_name == "mlp":
            fs_model = SafeMLPWrapper(batch_size=256,learning_rate_init=0.001, max_iter=100,n_repeats=2)
            print("Evaluating pure SFS : " + model_name)   

            sfs_start_time = perf_counter()
            sfs_selector = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('sfs', SequentialFeatureSelector(
                fs_model, 
                direction='forward',
                n_features_to_select='auto',
                tol=0.005,          # minimum R² gain
                scoring='r2', 
                cv=3, 
                n_jobs=-1
                ))
            ])
            sfs_selector.set_output(transform="pandas")
            #print(f"Running pure SFS for {model_name} (This may take a moment...)")
            sfs_selector.fit(X_train_auto_FULL, y_train_auto)
            sfs_features = sfs_selector.get_feature_names_out().tolist()
            print("Selected SFS columns:")
            print(sfs_features)
            sfs_end_time = perf_counter()

            with open(log_file_path, "a") as f:
                f.write("Evaluating sfs : " + model_name)
                for feature in sfs_features:
                    f.write(f"{feature},")
                f.write("\n")
            
            # Subset the unscaled data using the SFS selected features
            X_train_sfs = X_train_auto_FULL[sfs_features]
            # Preprocess test data
            preprocessor_test_sfs = Preprocessor(test_data, sfs_features)
            X_test_sfs, y_test_sfs, t_test_sfs, _ = preprocessor_test_sfs.preprocess_no_split()

            train_start_time = perf_counter()
            training_model = mlp_model
            builder_sfs = ModelBuilderMLP(X_train_sfs, X_test_sfs, y_train_auto, y_test_sfs, training_model, StandardScaler(),log_file_path=log_file_path)
            y_pred_sfs, learned_idle_power_sfs = builder_sfs.run_and_save_model(".", save=False)
            train_end_time = perf_counter()

            plotter_sfs = Plotter(y_pred_sfs, y_test_sfs, t_test_sfs)#, window_start =50, window_end=200)
            #plotter.plot_and_save("auto_gen_plots/", "pred_auto_" + PNG_NAME +'_' + model_name)
            plotter_sfs.plot_and_save("", "pred_sfs_" + PNG_NAME + '_' + model_name)

            sfs_execution_time = sfs_end_time - sfs_start_time
            training_execution_time = train_end_time - train_start_time
            print(f"SFS execution time: {sfs_execution_time:.2f} seconds\n")
            print(f"Model training execution time: {training_execution_time:.2f} seconds\n")
            with open(log_file_path, "a") as f:
                f.write(f"SFS execution time: {sfs_execution_time:.2f} seconds\n")
                f.write(f"Model training execution time: {training_execution_time:.2f} seconds\n")
                f.write(f"-------\n")      

        # else:
        #     num_features=len(features)
        #     fs_model = SafeKerasWrapper(dynamic_model(model_name,num_features,1))
        #     # Warning: CNN with Flatten does not work for SFS. 
        #     # CNN with GlobalAveragePool compiles with bad performance. 
        #     # Please comment it the out.
        #     print("Evaluating pure SFS : " + model_name)     
        #     sfs_start_time = perf_counter()
        #     sfs_selector = Pipeline(steps=[
        #         ('scaler', StandardScaler()),
        #         ('sfs', SequentialFeatureSelector(
        #         fs_model, 
        #         direction='forward',
        #         n_features_to_select='auto',
        #         tol=0.005,          # minimum R² gain
        #         scoring='r2', 
        #         cv=3, 
        #         n_jobs=-1
        #         ))
        #     ])
        #     sfs_selector.set_output(transform="pandas")
        #     #print(f"Running pure SFS for {model_name} (This may take a moment...)")
        #     sfs_selector.fit(X_train_auto_FULL, y_train_auto)
        #     sfs_features = sfs_selector.get_feature_names_out().tolist()
        #     print("Selected SFS columns:")
        #     print(sfs_features)
        #     sfs_end_time = perf_counter()
        #     # Subset the unscaled data using the SFS selected features
        #     X_train_sfs = X_train_auto_FULL[sfs_features]
        #     # Preprocess test data
        #     preprocessor_test_sfs = Preprocessor(test_data, sfs_features)
        #     X_test_sfs, y_test_sfs, t_test_sfs, _ = preprocessor_test_sfs.preprocess_no_split()
        #     num_features = len(sfs_features)
        #     train_start_time = perf_counter()
        #     training_model = dynamic_model(model_name,num_features,window_size)    
        #     builder_sfs = KerasModelBuilder(X_train_sfs, X_test_sfs, y_train_auto, y_test_sfs, training_model, StandardScaler(), 
        #                     window_size=window_size, train_epochs=30,file_written=True)
        #     y_pred_sfs, learned_idle_power_sfs = builder_sfs.run_and_save_model(".", save=False)
        #     train_end_time = perf_counter()

        #     plotter_sfs = Plotter(y_pred_sfs, y_test_sfs[window_size - 1:], t_test_sfs[window_size - 1:])
        #     #plotter.plot_and_save("auto_gen_plots/", "pred_auto_" + PNG_NAME +'_' + model_name)
        #     plotter_sfs.plot_and_save("", "pred_sfs_" + PNG_NAME + '_' + model_name)
        #     sfs_execution_time = sfs_end_time - sfs_start_time
        #     training_execution_time = train_end_time - train_start_time
        #     print(f"SFS execution time: {sfs_execution_time:.2f} seconds\n")
        #     print(f"Model training execution time: {training_execution_time:.2f} seconds\n")
        #     with open(log_file_path, "a") as f:
        #         f.write(f"SFS execution time: {sfs_execution_time:.2f} seconds\n")
        #         f.write(f"Model training execution time: {training_execution_time:.2f} seconds\n")
        #         f.write(f"-------\n") 


end_time = perf_counter()
total_execution_time = end_time - start_time
print(f"Total execution time: {total_execution_time:.2f} seconds\n")
with open(log_file_path, "a") as f:
    f.write(f"Total execution time: {total_execution_time:.2f} seconds\n")