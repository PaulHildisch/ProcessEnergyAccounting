import shap
import numpy as np
import pandas as pd
from model_testing.clean_impl.plotting.attributor_plotter import AttributionPlotter

class ProcessAttributorSHAP:

    def __init__(self,X_test, model, scaler):
        #X_test is scaled
        self.X_test = X_test
        self.model = model
        self.scaler = scaler
    

    def _init_explainer(self):
        idle_interval = np.zeros((1, self.X_test.shape[1]))
        idle_scaled = self.scaler.transform(idle_interval)

        #set the shapley baseline to idle instead of expected value
        self.explainer = shap.TreeExplainer(
            self.model, 
            data=idle_scaled, 
            feature_perturbation="interventional"
        )
        self.base_interval_energy = self.explainer.expected_value
        print(f"SHAP Base Power (Idle Baseline): {self.base_interval_energy:.2f} Ws")

    
    def attribute(self, df_original, good_features, test_times ,custom_name):
        self._init_explainer()

        #On some datasets, using this method with the GENERAL features creates a small additivity error < 3 Ws
        #Since the error we observed was small we deemed it acceptable 
        #When using the automatically selected features, this error also disappeared
        #We therefore strongly recommend using automatic selection when creating attributions
        shap_vals = self.explainer.shap_values(self.X_test, check_additivity=False)

        #In theory the shap values should be positive anyway since -> since SHAP values are calculated from the idle power upwards
        #Since we use an idle prediction this is not 100% correct, because the prediction will likely not be 100% equal with the true hidden idle state
        #This is a simplification, but it ensures there are no negative shap values
        #No process is able to create negative power
        shap_vals = np.maximum(0,shap_vals)
        df_budgets = pd.DataFrame(shap_vals, columns=good_features, index=test_times)      
        
        #Do we need this? -> Yes because of the different time zone on the server
        df_budgets.index = pd.to_datetime(df_budgets.index)
        if df_budgets.index.tz is None and df_original.index.tz is not None:
            print("actually aligned timezones" )
            df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        

        totals = df_original.groupby("_time")[good_features].sum()
        
        # Divide original metrics by the total to get the ratio (fillna(0) prevents division by zero)
        ratios = df_original[good_features].div(totals, axis=0).fillna(0)
        # Multiply ratios by the SHAP budgets and sum across the features to get energy per pid
        df_result = df_original.copy()
        df_result["attributed_dynamic_Wh"] = ratios.mul(df_budgets, axis=0).sum(axis=1)

        plotter = AttributionPlotter(df_result, time_col="_time", energy_col="attributed_dynamic_Wh")
        plotter.plot_top_processes(top_n=8, save_path=custom_name + "shap_process_attribution.png")
        plotter.plot_top_processes_by_max(top_n=8, save_path=custom_name + "shap_process_attribution_by_max.png")
        plotter.plot_top_processes_new(top_n=8, save_path=custom_name +"shap_process_attribution_new.png")
        plotter.plot_top_pids(top_n=8, save_path=custom_name+"shap_pid_attribution.png")
        
        return df_result


class ProcessAttributorEBM:

    def __init__(self,X_test, model, scaler):
        self.X_test = X_test
        self.model = model
        self.scaler = scaler
            

    def attribute(self, df_original, good_features, test_times, custom_name):
        
        #Create an idle baseline
        idle_scaled = self.scaler.transform(np.zeros((1, self.X_test.shape[1])))
        idle_scores = np.array(self.model.explain_local(idle_scaled).data(0)['scores'][:len(good_features)])

        #get the scores and subtract idle to show dynamic power
        #EBM is better suited for this task by design -> no additivity issues
        local_explain = self.model.explain_local(self.X_test)
        # adjusted_scores = [
        #     np.array(local_explain.data(i)['scores'][:len(good_features)]) - idle_scores
        #     for i in range(len(self.X_test))
        # ]

        adjusted_scores = []
        for i in range(len(self.X_test)):
            raw_adj = np.array(local_explain.data(i)['scores'][:len(good_features)]) - idle_scores
            clipped_adj = np.maximum(0, raw_adj)
            adjusted_scores.append(clipped_adj)


   
        df_budgets = pd.DataFrame(adjusted_scores, columns=good_features, index=test_times)
        df_budgets.index = pd.to_datetime(df_budgets.index)
        #adjust to server timezone
        if df_budgets.index.tz is None and df_original.index.tz is not None:
            df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        
        #create attribution ratios
        totals = df_original.groupby(level=0)[good_features].sum()
        ratios = df_original[good_features].div(totals, axis=0).fillna(0)
        
        df_result = df_original.copy()
        df_result["attributed_dynamic_Ws"] = ratios.mul(df_budgets, axis=0).sum(axis=1).values
        

        plotter = AttributionPlotter(df_result.reset_index(), time_col="_time", energy_col="attributed_dynamic_Ws")
        plotter.plot_top_processes(top_n=8, save_path=f"{custom_name}_ebm_process.png")
        plotter.plot_top_processes_by_max(top_n=8, save_path=f"{custom_name}_ebm_process_by_max.png")
        plotter.plot_top_processes_new(top_n=8, save_path=f"{custom_name}_ebm_process_new.png")
        plotter.plot_top_pids(top_n=8, save_path=f"{custom_name}_ebm_pid.png")

        
        return df_result