import shap
import numpy as np
import pandas as pd
from model_testing.clean_impl.plotting.attributor_plotter import AttributionPlotter

class ProcessAttributorSHAPMLP:

    def __init__(self,X_test, model, scaler):
        #X_test is scaled
        self.X_test = X_test
        self.model = model
        self.scaler = scaler
    

    def _init_explainer(self):
        idle_interval = np.zeros((1, self.X_test.shape[1]))
        idle_scaled = self.scaler.transform(idle_interval)

        #set the shapley basline to idle instead of expected value
        self.explainer = shap.KernelExplainer(
            self.model.predict, 
            data=idle_scaled, 
        )
        self.base_interval_energy = self.explainer.expected_value
        print(f"SHAP Base Power (Idle Baseline): {self.base_interval_energy:.2f} Ws")

    
    def attribute(self, df_original, good_features, test_times ,custom_name):
        self._init_explainer()
        shap_vals = self.explainer.shap_values(self.X_test)

        df_budgets = pd.DataFrame(shap_vals, columns=good_features, index=test_times)
        
        df_budgets.index = pd.to_datetime(df_budgets.index)
        if df_budgets.index.tz is None and df_original.index.tz is not None:
            print("actually aligned timezones" )
            df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        

        totals = df_original.groupby("_time")[good_features].sum()
        
        # Divide original metrics by the total to get the ratio (fillna(0) prevents division by zero)
        ratios = df_original[good_features].div(totals, axis=0).fillna(0)
        df_result = df_original.copy()
        df_result["attributed_dynamic_Ws"] = ratios.mul(df_budgets, axis=0).sum(axis=1)
        #print(df_result.head(5))

        #
        plotter = AttributionPlotter(df_result, time_col="_time", energy_col="attributed_dynamic_Ws")
        plotter.plot_top_processes(top_n=8, save_path=custom_name + "shap_process_attribution.png")
        #plotter.plot_top_processes_by_max(top_n=8, save_path=custom_name + "shap_process_attribution_by_max.png")
        plotter.plot_top_processes_new(top_n=8, save_path=custom_name +"shap_process_attribution_new.png")
        plotter.plot_top_pids(top_n=8, save_path=custom_name+"shap_pid_attribution.png")
        
        return df_result
