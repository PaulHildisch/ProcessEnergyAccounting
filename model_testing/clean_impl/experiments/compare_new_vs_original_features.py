import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#Custom imports
from model_testing.clean_impl.pipeline.model_builder import ModelBuilder
from model_testing.clean_impl.pipeline.wrappers import SafeEBMWrapper
from model_testing.clean_impl.pipeline.full_prediction_attribution_pipeline import  automatic_selection_prep, preprocess_test



#Full original feature set
original_features = [
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

#exclude 'avg_power',#'interval',# 'interval_energy',pid etc..
#temperature could be excluded because of inertia
new_features = [    
       'syscall_class_file', 'syscall_class_network',
       'context_switches',
       'delta_branch_instructions', 'delta_branch_misses',
       'delta_cache_misses', 'delta_cache_references', 'delta_cpu_migrations',
       'delta_cpu_ns', 'delta_cpu_time_proc', 'delta_cpu_time_psutil',
       'delta_cycles', 'delta_disk_read_bytes', 'delta_disk_write_bytes',
       'delta_dtlb_load_misses', 'delta_dtlb_store_misses', 'delta_fp_add_sub',
       'delta_fp_div', 'delta_fp_mac', 'delta_fp_mult', 'delta_instructions',
       'delta_io_bytes', 'delta_l1d_load_misses', 'delta_llc_load_misses',
       'delta_llc_store_misses', 'delta_net_recv_bytes',
       'delta_net_recv_packets', 'delta_net_send_bytes',
       'delta_net_send_packets', 'delta_node_load_misses',
       'delta_page_faults_maj', 'delta_page_faults_min',
       'delta_ref_cpu_cycles', 'delta_rss_memory',
       'delta_stalled_cycles_backend', 'delta_stalled_cycles_frontend',
       'hw_arch_arm64', 'hw_arch_other', 'hw_arch_riscv64', 'hw_arch_x86_64',
       'hw_core_count', 'hw_cores_17_32', 'hw_cores_1_4', 'hw_cores_33_plus',
       'hw_cores_5_8', 'hw_cores_9_16', 'hw_cores_unknown',
       'hw_cpu_governor_ondemand', 'hw_cpu_governor_performance',
       'hw_cpu_governor_powersave', 'hw_cpu_governor_schedutil',
       'hw_cpu_governor_unknown', 'hw_cpu_vendor_amd', 'hw_cpu_vendor_apple',
       'hw_cpu_vendor_arm', 'hw_cpu_vendor_intel', 'hw_cpu_vendor_other',
       'hw_fan_count', 'hw_fans_0', 'hw_fans_1', 'hw_fans_2_plus',
       'hw_fans_unknown', 'hw_freq_ratio', 'hw_numa_node_count',
       'hw_ram_129gb_plus', 'hw_ram_16_32gb', 'hw_ram_33_64gb',
       'hw_ram_65_128gb', 'hw_ram_lt16gb', 'hw_ram_slot_count',
       'hw_ram_slots_dual', 'hw_ram_slots_quad_or_more', 'hw_ram_slots_single',
       'hw_ram_slots_unknown', 'hw_ram_total_gb', 'hw_ram_unknown',
       'hw_tdp_tier_high', 'hw_tdp_tier_low', 'hw_tdp_tier_mid',
       'hw_tdp_tier_unknown', 'hw_temp_cool', 'hw_temp_hot', 'hw_temp_normal',
       'hw_temp_unknown', 'hw_temperature_c',
       'syscall_class_other', 'syscall_count', 'syscall_class_memory',
       'syscall_class_process', 'syscall_class_signal', 'syscall_class_sched',
       'syscall_class_time']


def select_new_feat_data(dataset_name):
    
    #Recorded with an extended feature set | is used for the feature comparison expermiment
    if dataset_name == "AMPLISEQ_S12_NEW_FEAT":
        train_workflows = [
            pd.read_parquet("runs/new_feature_siena12/ampliseq1_new_feat.parquet"),
            pd.read_parquet("runs/new_feature_siena12/ampliseq2_new_feat.parquet"),
            pd.read_parquet("runs/new_feature_siena12/ampliseq3_new_feat.parquet")
        ]
        test_workflows = pd.read_parquet("runs/new_feature_siena12/ampliseq4_new_feat.parquet")

    #Recorded with an extended feature set | | is used for the feature comparison expermiment
    elif dataset_name == "SAREK_S12_NEW_FEAT":
        train_workflows = [
                pd.read_parquet("runs/new_feature_siena12/sarek1_new_feat.parquet"),
                pd.read_parquet("runs/new_feature_siena12/sarek2_new_feat.parquet")
        ]
        test_workflows = pd.read_parquet("runs/new_feature_siena12/sarek3_new_feat.parquet")

    elif dataset_name == "MIXED_UNKNOWN_TYPE_S12_NEW_FEAT":
        train_workflows =[
            pd.read_parquet("runs/new_feature_siena12/rnaseq1_new_feat.parquet"),
            pd.read_parquet("runs/new_feature_siena12/ampliseq1_new_feat.parquet")
        ]
        test_workflows = pd.read_parquet("runs/new_feature_siena12/sarek1_new_feat.parquet")

    elif dataset_name == "MIXED_KNOWN_TYPE_S12_NEW_FEAT":
        train_workflows =[
            pd.read_parquet("runs/new_feature_siena12/rnaseq1_new_feat.parquet"),
            pd.read_parquet("runs/new_feature_siena12/ampliseq1_new_feat.parquet"),
            pd.read_parquet("runs/new_feature_siena12/sarek1_new_feat.parquet")
        ]
        test_workflows = pd.read_parquet("runs/new_feature_siena12/sarek2_new_feat.parquet")

    else:
        raise ValueError("UNKOWN DATASET SELECTED! Choose valide name.")

    if len(train_workflows) > 1:
        training_data = pd.concat(train_workflows, ignore_index=True)
    else:
        training_data = train_workflows[0]

    test_data = test_workflows
    return training_data, test_data



def experiment(model, dataset_name, full_features):
    training_data, test_data = select_new_feat_data(dataset_name)
    selected_features,X_train, y_train =  automatic_selection_prep(training_data, full_features, model)
    X_test, y_test, t_test , X_test_unaggregated = preprocess_test(test_data ,selected_features)
    builder = ModelBuilder(X_train, X_test, y_train, y_test, model, StandardScaler())
    #prints the evaluation but will not generate plots
    y_pred, learned_idle_power = builder.run_and_save_model(".", model_name="new_vs_old_feat_model.joblib", save=False)
    return selected_features


def evaluate(model, data):
    print("Using original feature set: ")
    og_selected = experiment(model, data, original_features)
    print("Using new feature set: ")
    new_selected = experiment(model, data, new_features)
    print("Only in original set:")
    print(set(og_selected) - set(new_selected))

    print("Only in new set: ")
    print(set(new_selected)- set(og_selected))

    print("Present in both: ")
    print(set(new_selected).intersection(set(og_selected)))
    print('-'*150)

if __name__ == "__main__":
    #Only choose Rf for the paper due to space constraints
    model = RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42)
    #model = SafeEBMWrapper()
    #model = Ridge(alpha=1.0)
    #model = Lasso(alpha=0.1)
    print("Analyze SAREK new features")
    evaluate(model, "SAREK_S12_NEW_FEAT")

    print("Analyze AMPLISEQ new features")
    evaluate(model, "AMPLISEQ_S12_NEW_FEAT")

    print("Analyze MIXED UNKNOWN new features")
    evaluate(model, "MIXED_UNKNOWN_TYPE_S12_NEW_FEAT")

    print("Analyze MIXED KOWN new features")
    evaluate(model, "MIXED_KNOWN_TYPE_S12_NEW_FEAT")
