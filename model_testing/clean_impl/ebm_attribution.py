#Modified for EBM from shapley_improved.py

import numpy as np
import pandas as pd
from attributor_plotter import AttributionPlotter

class ProcessAttributorEBM:

    def __init__(self,X_test, model, scaler):
        self.X_test = X_test
        self.model = model
        self.scaler = scaler
    
    def attribute(self, df_original, good_features, test_times):
        
        local_explain = self.model.explain_local(self.X_test)
        atrr_val = []
        for i in range(len(self.X_test)):
            row_data = local_explain.data(i)
            #print(row_data['values'])
            atrr_val.append(row_data['values'][:len(good_features)])
        atrr_val = np.array(atrr_val)

        #This is questionable due to the idle baseline there should only be postivite attributions?
        #shap_vals = np.maximum(0,shap_vals)
        df_budgets = pd.DataFrame(atrr_val, columns=good_features, index=test_times)

        #Do we need this? -> Yes because of the different time zone on the server
        df_budgets.index = pd.to_datetime(df_budgets.index)
        if df_budgets.index.tz is None and df_original.index.tz is not None:
            print("actually aligned timezones" )
            df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        
        # Divide original metrics by the total to get the ratio (fillna(0) prevents division by zero)
        ratios = df_original[good_features].div(df_budgets, axis=0).fillna(0)
        
        # Multiply ratios by the budgets, then sum across the features (axis=1) to get final 
        df_result = df_original.copy()
        df_result["attributed_dynamic_Ws"] = ratios.mul(df_budgets, axis=0).sum(axis=1)
        print(df_result.head(5))

        #
        plotter = AttributionPlotter(df_result, time_col="_time", energy_col="attributed_dynamic_Ws")
        plotter.plot_top_processes(top_n=8, save_path="ebm_process_attribution.png")
        plotter.plot_top_processes_new(top_n=8, save_path="ebm_process_attribution_new.png")
        plotter.plot_top_pids(top_n=8, save_path="ebm_pid_attribution.png")
        
        return df_result

        
       