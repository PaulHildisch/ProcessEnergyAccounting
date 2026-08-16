#Compiling:
#.../ProcessEnergyAccounting$ python3 -m model_testing.clean_impl.pipeline.full_prediction_attribution_pipeline_dl
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from keras import layers, optimizers, callbacks, Sequential
import os

#Custom imports
from model_testing.clean_impl.pipeline.preprocessing import Preprocessor
from model_testing.clean_impl.pipeline.model_builder import ModelBuilder
from model_testing.clean_impl.pipeline.model_builder_keras import KerasModelBuilder

from model_testing.clean_impl.plotting.plotting import Plotter
from model_testing.clean_impl.pipeline.universal_filtering import CustomSpearmanFilter
from model_testing.clean_impl.pipeline.wrappers_dl import SafeMLPWrapper, SafeKerasWrapper

from model_testing.clean_impl.pipeline.shapley_improved_dl import ProcessAttributorSHAPMLP


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Full feature set
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

general_features =  ['delta_io_bytes', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']
#other_path = "../../ProcessEnergyAccounting/"
other_path = ""
def select_data(dataset_name):
    if dataset_name == "AMPLISEQ":
        train_workflows = [
                pd.read_parquet(other_path+"runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
                pd.read_parquet(other_path+"runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
                pd.read_parquet(other_path+"runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet")
        ]
        test_workflows = pd.read_parquet(other_path+"runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet")

    elif dataset_name == "SAREK":
        train_workflows = [
            pd.read_parquet(other_path+"runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
            pd.read_parquet(other_path+"runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet")

        ]
        test_workflows = pd.read_parquet(other_path+"runs/nfcore-20260708T212252Z/datasets/sarek3_0907.parquet")

    elif dataset_name == "MIXED_UNKOWN_TYPE":
        #Be careful with uncommenting: train data must not contain test data
        train_workflows = [
            pd.read_parquet(other_path+"runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
            pd.read_parquet(other_path+"runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
            #pd.read_parquet(other_path+"runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
            pd.read_parquet(other_path+"runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")

        ]
        #test_mixed_unseen_type = pd.read_parquet(other_path+"runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")
        test_workflows = pd.read_parquet(other_path+"runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet")

    #CPU06 data is more relevant to auto vs generalized and the cross node comparison, therefore not all data sets are integrated here
    elif dataset_name == "AMPLISEQ_CPU06":
        train_workflows = [
            pd.read_parquet("runs/cpu06_scondtry/ampliseq-1786576997.parquet")
                ]
        test_workflows = pd.read_parquet("runs/cpu06_scondtry/ampliseq-1786566355.parquet")

    #only local debugging
    elif dataset_name == "DEBUG_LOCAL":
        train_workflows = [
                        pd.read_parquet(other_path+"data/siena12/test/sarek_1.parquet")
                ]
        test_workflows = pd.read_parquet(other_path+"data/siena12/test/sarek_2.parquet")
        
    else:
        raise ValueError("UNKOWN DATASET SELECTED! Choose valide name.")

    if len(train_workflows) > 1:
        training_data = pd.concat(train_workflows, ignore_index=True)
    else:
        training_data = train_workflows[0]

    test_data = test_workflows
    return training_data, test_data


def automatic_selection_prep(training_data, full_features, model, small_pipiline_used=False):
    preprocessor_train = Preprocessor(training_data, full_features)
    X_train_FULL, y_train, t_train, _ = preprocessor_train.preprocess_no_split()

    automatic_feature_selection = Pipeline(steps=[
            ('variance', VarianceThreshold(threshold=0.01)),
            ('decorrelate', CustomSpearmanFilter(threshold=0.80)),
            ('scaler', StandardScaler()),
            ('select_features', SelectFromModel(model, threshold='0.5*median'))
        ])
    automatic_feature_selection_small = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('select_features', SelectFromModel(model, threshold='0.5*median'))
        ])
    if small_pipiline_used:
        automatic_feature_selection=automatic_feature_selection_small
    automatic_feature_selection.set_output(transform="pandas")
    automatic_feature_selection.fit_transform(X_train_FULL, y_train)
    good_features = automatic_feature_selection.get_feature_names_out().tolist()
    X_train = X_train_FULL[good_features]

    print("Selected columns:")
    print(good_features)
    return good_features, X_train, y_train

def general_feature_prep(training_data, genereal_features):
    preprocessor_train = Preprocessor(training_data, general_features)
    X_train, y_train, t_train, _ = preprocessor_train.preprocess_no_split()
    return general_features, X_train, y_train

def preprocess_test(test_data, selected_features):
    preprocessor_test = Preprocessor(test_data, selected_features)
    #t_test only need for plot_data
    X_test, y_test, t_test , X_test_unaggregated = preprocessor_test.preprocess_no_split()
    return X_test, y_test, t_test , X_test_unaggregated

#In opposite to full_prediction_attribution_pipeline.py, pass here the model name.
def pipeline(mode, full_features, general_features, model_name, dataset_name, attribute=True):
    training_data, test_data = select_data(dataset_name)
    #Does the model need to be copied?
    
    if mode == "AUTO":
        num_features=len(full_features)
        small_pipiline_used=False
        fs_model = None
        if model_name == "mlp":
            fs_model = SafeMLPWrapper(batch_size=256)
        else:
            fs_model = SafeKerasWrapper(dynamic_model(model_name,num_features,1))
            small_pipiline_used=True

        selected_features,X_train, y_train = automatic_selection_prep(training_data, full_features, fs_model, small_pipiline_used)
    elif mode == "GENERAL":
        selected_features,X_train, y_train = general_feature_prep(training_data, general_features)
    else:
        raise ValueError("Unallowed mode selected!")

    X_test, y_test, t_test , X_test_unaggregated = preprocess_test(test_data,selected_features)
    window_size = 20 # choose window size > 1 for context based model such as CNN
    if model_name == "mlp":
        training_model = mlp_model
        builder = ModelBuilder(X_train, X_test, y_train, y_test, training_model, StandardScaler())  
    else:
        num_features = len(selected_features)
        
        training_model = dynamic_model(model_name,num_features,window_size)  
        builder= KerasModelBuilder(X_train, X_test, y_train, y_test, training_model, StandardScaler(), 
           window_size=window_size, train_epochs=30)              
        
    #Attributors recalculate idle power anyway
    y_pred, learned_idle_power = builder.run_and_save_model(".", model_name="full_pipeline_model.joblib", save=True)

    #Plot prediction results
    if model_name == "mlp" or window_size==1:
        plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
    else:
        plotter = Plotter(y_pred,y_test[window_size - 1:], t_test[window_size - 1:])

    plotter.plot_and_save("", "actual_energy_vs_predicted")
    #Attribute prediction
    if attribute:
        if model_name == "mlp":
            attributor = ProcessAttributorSHAPMLP( builder.X_test_scaled, builder.model, builder.scaler)
            attributor.attribute(X_test_unaggregated,selected_features,t_test.values , "MLP_")
        else:
            print("Attribution for this model type is not yet supported. Please choose only MLP Attribution.")
    else:
        print("Skipping attribution was selected!")

#Choose other Keras model if required.
def dynamic_model(model_name, num_features, window_size):
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

if __name__ == "__main__":

    #pipeline("GENERAL", features, general_features, "cnn", "MIXED_UNKOWN_TYPE", attribute=False)
    pipeline("AUTO", features, general_features, "mlp", "AMPLISEQ", attribute=True)