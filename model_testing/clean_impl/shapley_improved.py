import shap
import numpy as np
import pandas as pd
from attributor_plotter import AttributionPlotter

class ProcessAttributorSHAP:

    def __init__(self,X_test, model, scaler):
        self.X_test = X_test
        self.model = model
        self.scaler = scaler
    

    def _init_explainer(self):
        idle_interval = np.zeros((1, self.X_test.shape[1]))
        idle_scaled = self.scaler.transform(idle_interval)

        #set the shapley basline to idle instead of expected value
        self.explainer = shap.TreeExplainer(
            self.model, 
            data=idle_scaled, 
            feature_perturbation="interventional"
        )
        self.base_interval_energy = self.explainer.expected_value
        print(f"SHAP Base Power (Idle Baseline): {self.base_interval_energy:.2f} Ws")

    
    def attribute(self, df_original, good_features, test_times):
        self._init_explainer()
        
        # 1. Generate & Clip SHAP Budgets (Numpy is faster)
        shap_vals = self.explainer.shap_values(self.X_test)

        #This is questionable due to the idle baseline there should only be postivite attributions?
        shap_vals = np.maximum(0,shap_vals)
        df_budgets = pd.DataFrame(shap_vals, columns=good_features, index=test_times)
        print(df_budgets.head(10))
        
        #Do we need this?
        df_budgets.index = pd.to_datetime(df_budgets.index)
        if df_budgets.index.tz is None and df_original.index.tz is not None:
            print("actually aligned timezones" )
            df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        

        #Check this part and also check the "original df?"
        # 2. Pure Pandas Vectorization (No loops!)
        # Assuming df_original index is time (or groupby "_time")
        totals = df_original.groupby("_time")[good_features].sum()
        
        # Divide original metrics by the total to get the ratio (fillna(0) prevents division by zero)
        ratios = df_original[good_features].div(totals, axis=0).fillna(0)
        
        # Multiply ratios by the SHAP budgets, then sum across the features (axis=1) to get final Wh
        df_result = df_original.copy()
        df_result["attributed_dynamic_Wh"] = ratios.mul(df_budgets, axis=0).sum(axis=1)

        # 3. Plotting
        plotter = AttributionPlotter(df_result, time_col="_time", energy_col="attributed_dynamic_Wh")
        plotter.plot_top_processes(top_n=8, save_path="shap_process_attribution.png")
        plotter.plot_top_processes_new(top_n=8, save_path="shap_process_attribution_new.png")
        plotter.plot_top_pids(top_n=8, save_path="shap_pid_attribution.png")
        
        return df_result
    