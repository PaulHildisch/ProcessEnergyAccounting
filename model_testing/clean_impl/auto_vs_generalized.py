from model_builder import ModelBuilder
from preprocessing import Preprocessor
from plotting import Plotter
from plotting import plot_dataset
#from shapley import ProcessAttributor
from model_testing.clean_impl.shapley_improved_old import ProcessAttributorSHAP
from model_testing.clean_impl.shapley_improved_old import ProcessAttributorEBM

from universal_filtering import CustomSpearmanFilter
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV

# Define a grid of priors to test
# 1e-6 is the default (uninformative), 1e-2 is a stronger prior
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from interpret.glassbox import ExplainableBoostingRegressor

#create thsi wrapper to be able to use with scikit learn
class SafeEBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, interactions=2, max_rounds=2000):
        self.interactions = interactions
        self.max_rounds = max_rounds
        self.model = None

    def fit(self, X, y):
        self.model = ExplainableBoostingRegressor(
            interactions=self.interactions,
            max_rounds=self.max_rounds,
            n_jobs=-1,
            random_state=42
        )
        
        self.model.fit(X, y)
        n_features = X.shape[1]
        all_importances = self.model.term_importances()
        
        # Now convert to numpy array and slice it for SelectFromModel
        self.feature_importances_ = np.array(all_importances)[:n_features]
        return self

    def predict(self, X):
        return self.model.predict(X)

#soiwe die original e lasso
class CvxpyMimicLasso(BaseEstimator, RegressorMixin):
    def __init__(self, l1_penalty=0.1):
        self.l1_penalty = l1_penalty
        self.model = None

    def fit(self, X, y):

        N = X.shape[0]
        sklearn_alpha = self.l1_penalty / (2 * N)
    
        self.model = Lasso(
            alpha=sklearn_alpha, 
            positive=True, 
            fit_intercept=True, 
            max_iter=10000
        )
        self.model.fit(X, y)
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self

    def predict(self, X):
        return self.model.predict(X)



train_ampliseq = [
        pd.read_parquet("runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
        pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
        pd.read_parquet("runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet")

]

test_ampliseq = pd.read_parquet("runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet")




train_sarek = [
    pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
    pd.read_parquet("runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet")

]

test_sarek = pd.read_parquet("runs/nfcore-20260708T212252Z/datasets/sarek3_0907.parquet")



train_mixed_unseen_type2 = [
    pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
    pd.read_parquet("runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
    #pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
    pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")

]
#test_mixed_unseen_type = pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")
test_mixed_unseen_type2 = pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet")




test_stressng = pd.read_parquet("stressng_test3_10_.parquet")
train_stressng = pd.read_parquet("stressng_train0_3_.parquet")



data_map = {
    "ampliseq": (train_ampliseq,test_ampliseq),
    #"sarek" : (train_sarek, test_sarek),
    #"train_mixed_unseen_type2": (train_mixed_unseen_type2,test_mixed_unseen_type2)
}
#TODO add better stressng

# test_long_on_short_training = pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")
# short_training = [
#     pd.read_parquet("runs/nfcore-20260630T142308Z/datasets/rnaseq1_shorttest.parquet"),
#     pd.read_parquet("runs/nfcore-20260630T143512Z/datasets/chipseq1_shorttest.parquet"),
#     pd.read_parquet("runs/nfcore-20260630T152039Z/datasets/methlyseq1_shorttest.parquet"),
#     # pd.read_parquet("runs/nfcore-20260630T152447Z/datasets/methylseq2_shorttest.parquet"),
#     pd.read_parquet("runs/nfcore-20260630T153034Z/datasets/sarek1_shorttest.parquet")
#     # pd.read_parquet("runs/nfcore-20260630T153034Z/datasets/sarek1_shorttest.parquet"),
#     # pd.read_parquet("runs/nfcore-20260630T153801Z/datasets/sarek2_short_test.parquet"),
#     #pd.read_parquet("data/siena12/stressng_siena12.parquet")]
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

#TODO two missing for workflows
generalized_features =  ['delta_io_bytes', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']
#generalized_features =  ['delta_io_bytes', 'context_switches', 'delta_cpu_ns',  'syscall_count']
#Workflow gen -> pretty good
#generalized_features =['delta_io_bytes', 'syscall_class_network', 'syscall_class_memory', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']
for name ,value in data_map.items():

    training_data = value[0]
    training_data = pd.concat(training_data, ignore_index=True)
    test_data = value[1]
    PNG_NAME = name
    print("/n")
    print("Evaluating : ", name)



    preprocessor_train = Preprocessor(training_data, generalized_features)
    X_train, y_train, t_train, _ = preprocessor_train.preprocess_no_split()
    #plot_dataset(t_train, y_train, "multi_training_gen_" +name)

    preprocessor_test = Preprocessor(test_data, generalized_features)
    X_test, y_test, t_test , X_test_unaggregated = preprocessor_test.preprocess_no_split()
    #plot_dataset(t_test, y_test, "multi_testing_gen_"+name)

    preprocessor_train_auto = Preprocessor(training_data, features)
    X_train_auto_FULL, y_train_auto, t_train_auto, _ = preprocessor_train_auto.preprocess_no_split()
    plot_dataset(t_train_auto, y_train_auto, "multi_training_auto_" +name)


    


    
    models = {
        "RF": RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42),
        #"SgD" :SGDRegressor(loss= "squared_error", penalty='l2', shuffle= False),
        #"Ridge" : Ridge(alpha=1.0),
        #"Lasso" : Lasso(alpha=0.1), # do this with their implementation |
        #"Lasso_Cvxpy": CvxpyMimicLasso(l1_penalty=0.1),
        #"bayes" : linear_model.BayesianRidge(),
        "EBM" : SafeEBMWrapper()

        #"kernelRidge" : KernelRidge(alpha=1.0) this doesnt work like that with the selection
    }

    #idle_power_isactually idle interval energy
    for model_name, model in models.items():
        print("Evaluating gen : " + model_name)
        print(generalized_features)
        builder = ModelBuilder(X_train, X_test, y_train, y_test, clone(model), StandardScaler())
        y_pred, learned_idle_power = builder.run_and_save_model(".")   

        # plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
        # plotter.plot_and_save("auto_gen_plots/", "pred_gen_" + PNG_NAME +'_' + model_name)


        
        print("Evaluating auto : " + model_name)     
        automatic_feature_selection = Pipeline(steps=[
            ('variance', VarianceThreshold(threshold=0.01)), #explain this
            ('decorrelate', CustomSpearmanFilter(threshold=0.80)),
            ('scaler', StandardScaler()),
            ('select_features', SelectFromModel(clone(model), threshold='0.5*median'))
        ])

        automatic_feature_selection.set_output(transform="pandas")
        #X_train_auto = automatic_feature_selection.fit_transform(X_train_auto, y_train_auto)
        automatic_feature_selection.fit_transform(X_train_auto_FULL, y_train_auto)
        good_features = automatic_feature_selection.get_feature_names_out().tolist()
        print("Selected auto columns:")
        print(good_features)
        X_train_auto = X_train_auto_FULL[good_features]


        preprocessor_test_auto = Preprocessor(test_data, good_features)
        X_test_auto, y_test_auto, t_test_auto , X_test_unaggregated_auto = preprocessor_test_auto.preprocess_no_split()
        #plot_dataset(t_test_auto, y_test_auto, "multi_testing_auto_"+name)
        builder_auto = ModelBuilder(X_train_auto, X_test_auto, y_train_auto, y_test_auto, clone(model), StandardScaler())
        y_pred_auto, learned_idle_power_auto = builder_auto.run_and_save_model(".")

        # plotter = Plotter(y_pred_auto,y_test_auto, t_test_auto)#, window_start =50, window_end=200)
        # plotter.plot_and_save("auto_gen_plots/", "pred_auto_" + PNG_NAME +'_' + model_name)






        #unccometn for sfs
        # print("Evaluating pure SFS : " + model_name)     
        # sfs_selector = Pipeline(steps=[
        #     ('scaler', StandardScaler()),
        #     ('sfs', SequentialFeatureSelector(
        #         clone(model), 
        #         direction='forward',
        #         n_features_to_select='auto',
        #         tol=0.005,                 # Minimum R2 gain
        #         scoring='r2', 
        #         cv=3, 
        #         n_jobs=-1
        #     ))
        # ])

        # sfs_selector.set_output(transform="pandas")
        # #print(f"Running pure SFS for {model_name} (This may take a moment...)")
        # sfs_selector.fit(X_train_auto_FULL, y_train_auto)
        # sfs_features = sfs_selector.get_feature_names_out().tolist()
        # print("Selected SFS columns:")
        # print(sfs_features)
        
        # # Subset the unscaled data using the SFS selected features
        # X_train_sfs = X_train_auto_FULL[sfs_features]
        # # Preprocess test data
        # preprocessor_test_sfs = Preprocessor(test_data, sfs_features)
        # X_test_sfs, y_test_sfs, t_test_sfs, _ = preprocessor_test_sfs.preprocess_no_split()
        
        # builder_sfs = ModelBuilder(
        #     X_train_sfs, 
        #     X_test_sfs, 
        #     y_train_auto, 
        #     y_test_sfs, 
        #     clone(model), 
        #     StandardScaler()
        # )
        # y_pred_sfs, learned_idle_power_sfs = builder_sfs.run_and_save_model(".")

        # # Plot and save
        # plotter_sfs = Plotter(y_pred_sfs, y_test_sfs, t_test_sfs)
        # plotter_sfs.plot_and_save("auto_gen_plots/", "pred_sfs_" + PNG_NAME + '_' + model_name)
        if model_name == "RF":
            attributor_auto = ProcessAttributorSHAP( builder_auto.X_test_scaled, builder_auto.model, builder_auto.scaler)
            attributor_auto.attribute(X_test_unaggregated_auto,good_features,t_test.values,"attributions/"+name+ model_name+ "_auto_")

        if model_name == "EBM":
            #Acess true model from the wrapper
            attributor_auto = ProcessAttributorEBM( builder_auto.X_test_scaled, builder_auto.model.model, builder_auto.scaler)
            attributor_auto.attribute(X_test_unaggregated_auto,good_features,t_test.values,"attributions/"+name+ model_name+ "_auto_")

            
            #attribution gen vs auto -> auto is better
            # attributor = ProcessAttributorSHAP( builder.X_test_scaled, builder.model, builder.scaler)
            # attributor.attribute(X_test_unaggregated,good_features,t_test.values, model_name+ "_gen_")
        # else:
        #     attributor_auto = ProcessAttributorLinear(  builder_auto.model, builder_auto.scaler)
        #     attributor_auto.attribute(X_test_unaggregated_auto,good_features,t_test.values,"attributions/"+name + model_name+ "_auto_")


#Nur zum Test -> eigentlich Pauls Aufgabe
# attributor = ProcessAttributorLinear(  builder.model, builder.scaler)
# attributor.attribute(X_test_unaggregated,good_features,t_test.values)
        print()
        print()





#check if we ann pass this differently
#attributor = ProcessAttributorSHAP( builder.X_test_scaled, builder.model, builder.scaler)
#attributor.attribute(X_test_unaggregated,good_features,t_test.values)


#Nur zum Test -> eigentlich Pauls Aufgabe
# attributor = ProcessAttributorLinear(  builder.model, builder.scaler)
# attributor.attribute(X_test_unaggregated,good_features,t_test.values)



