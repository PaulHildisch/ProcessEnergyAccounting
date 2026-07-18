import shap
import numpy as np
import pandas as pd
from attributor_plotter import AttributionPlotter

class ProcessAttributorSHAP:

    def __init__(self,X_test, model, scaler):
        #X_test is scaled
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

    
    def attribute(self, df_original, good_features, test_times ,custom_name):
        self._init_explainer()
        shap_vals = self.explainer.shap_values(self.X_test)
        #Look into learning constraints?
        #This is questionable due to the idle baseline there should only be postivite attributions?
        #There is no visible difference in the plots -> so the difference is small at most 
        #In theory the shap values should be positive anyway since -> since they are calculated from the idle power upwards
        #shap_vals = np.maximum(0,shap_vals)
        df_budgets = pd.DataFrame(shap_vals, columns=good_features, index=test_times)
        #print(df_budgets.head(10))
        
        #Do we need this? -> Yes because of the different time zone on the server
        df_budgets.index = pd.to_datetime(df_budgets.index)
        if df_budgets.index.tz is None and df_original.index.tz is not None:
            print("actually aligned timezones" )
            df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        

        totals = df_original.groupby("_time")[good_features].sum()
        
        # Divide original metrics by the total to get the ratio (fillna(0) prevents division by zero)
        ratios = df_original[good_features].div(totals, axis=0).fillna(0)
        # Multiply ratios by the SHAP budgets, then sum across the features (axis=1) to get final 
        df_result = df_original.copy()
        df_result["attributed_dynamic_Wh"] = ratios.mul(df_budgets, axis=0).sum(axis=1)
        #print(df_result.head(5))

        #
        plotter = AttributionPlotter(df_result, time_col="_time", energy_col="attributed_dynamic_Wh")
        plotter.plot_top_processes(top_n=8, save_path=custom_name + "shap_process_attribution.png")
        plotter.plot_top_processes_new(top_n=8, save_path=custom_name +"shap_process_attribution_new.png")
        plotter.plot_top_pids(top_n=8, save_path=custom_name+"shap_pid_attribution.png")
        
        return df_result



#Don't use this, this has issues !
# class ProcessAttributorLinear:
#     def __init__(self, model, scaler):
#         self.model = model
#         self.scaler = scaler

#     def attribute(self, df_original, good_features, test_times, custom_name):
#         print("Extracting linear weights for attribution...")
#         df_result = df_original.copy()
        

#         weights = self.model.coef_
#         if hasattr(self.scaler, 'scale_'):
#             effective_weights = weights / self.scaler.scale_
#         else:
#             effective_weights = weights
            

#         df_result["attributed_dynamic_Wh"] = df_result[good_features].values @ effective_weights
#         df_result["base_name"] = (
#             df_result["process_name"].str.replace(r"_\d+$", "", regex=True).str.strip()
#         )
#         df_result.loc[df_result["base_name"] == "", "base_name"] = "unknown"

#         #Aggregate and Pivot
#         agg = (
#             df_result.groupby(["_time", "base_name"])["attributed_dynamic_Wh"]
#             .sum()
#             .reset_index()
#         )
        
#         pivot = agg.pivot(
#             index="_time", columns="base_name", values="attributed_dynamic_Wh"
#         ).fillna(0)

#         N = 8
#         top_processes = pivot.max().sort_values(ascending=False).head(N).index
#         pivot_top = pivot[top_processes].copy()

#         if len(pivot.columns) > N:
#             pivot_top["Other"] = pivot.drop(columns=top_processes).sum(axis=1)


#         neg_count = (pivot_top < 0).sum().sum()
#         if neg_count > 0:
#             print(f"Warning: Found {neg_count} minor negative values (likely float precision). Clipping to 0.")
        
     
#         pivot_top_clipped = pivot_top.clip(lower=0)
        
#         # Filter for the test times provided by the pipeline
#         pivot_mask = pivot_top_clipped.index.isin(test_times)
#         final_pivot = pivot_top_clipped[pivot_mask]



# #this is probably wrong
# #Check this right here
#         plotter = AttributionPlotter(df_result, time_col="_time", energy_col="attributed_dynamic_Wh")
#         plotter.plot_top_processes(top_8, save_path="l_process_attribution.png")

#         df_result["process_name"] = df_result["base_name"]

#         plotter = AttributionPlotter(df_result, time_col="_time", energy_col="attributed_dynamic_Wh")

#         plotter.plot_top_processes(top_n=8, save_path=custom_name+"_linear_process_attribution.png")
#         plotter.plot_top_processes_new(top_n=8, save_path=custom_name +"linear_process_attribution_new.png")
#         plotter.plot_top_pids(top_n=8, save_path=custom_name+"linear_pid_attribution.png")
        
#         return df_result, final_pivot




import numpy as np
import pandas as pd
from attributor_plotter import AttributionPlotter

class ProcessAttributorEBM:

    def __init__(self,X_test, model, scaler):
        self.X_test = X_test
        self.model = model
        self.scaler = scaler
    
    # def attribute(self, df_original, good_features, test_times, custom_name):
        
    #     local_explain = self.model.explain_local(self.X_test)
    #     atrr_val = []
    #     for i in range(len(self.X_test)):
    #         row_data = local_explain.data(i)
    #         #print(row_data['values'])
    #         atrr_val.append(row_data['values'][:len(good_features)])
    #     atrr_val = np.array(atrr_val)

    #     #This is questionable due to the idle baseline there should only be postivite attributions?
    #     #shap_vals = np.maximum(0,shap_vals)
    #     df_budgets = pd.DataFrame(atrr_val, columns=good_features, index=test_times)

    #     #Do we need this? -> Yes because of the different time zone on the server
    #     df_budgets.index = pd.to_datetime(df_budgets.index)
    #     if df_budgets.index.tz is None and df_original.index.tz is not None:
    #         print("actually aligned timezones" )
    #         df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        
    #     # Divide original metrics by the total to get the ratio (fillna(0) prevents division by zero)
    #     ratios = df_original[good_features].div(df_budgets, axis=0).fillna(0)
        
    #     # Multiply ratios by the budgets, then sum across the features (axis=1) to get final 
    #     df_result = df_original.copy()
    #     df_result["attributed_dynamic_Ws"] = ratios.mul(df_budgets, axis=0).sum(axis=1)
    #     print(df_result.head(5))

    #     #
    #     plotter = AttributionPlotter(df_result, time_col="_time", energy_col="attributed_dynamic_Ws")
    #     plotter.plot_top_processes(top_n=8, save_path=custom_name + "ebm_process_attribution.png")
    #     plotter.plot_top_processes_new(top_n=8, save_path=custom_name + "ebm_process_attribution_new.png")
    #     plotter.plot_top_pids(top_n=8, save_path=custom_name + "ebm_pid_attribution.png")
        
    #     return df_result
        

    def attribute(self, df_original, good_features, test_times, custom_name):
        
        #Create an idle baseline
        idle_scaled = self.scaler.transform(np.zeros((1, self.X_test.shape[1])))
        idle_scores = np.array(self.model.explain_local(idle_scaled).data(0)['scores'][:len(good_features)])

        #get the scores and subtract idle to show dynamic power
        local_explain = self.model.explain_local(self.X_test)
        adjusted_scores = [
            np.array(local_explain.data(i)['scores'][:len(good_features)]) - idle_scores
            for i in range(len(self.X_test))
        ]

   
        df_budgets = pd.DataFrame(adjusted_scores, columns=good_features, index=test_times)
        df_budgets.index = pd.to_datetime(df_budgets.index)
        #watch out for server timezone
        if df_budgets.index.tz is None and df_original.index.tz is not None:
            df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        
        #create attribution ratios
        totals = df_original.groupby(level=0)[good_features].sum()
        ratios = df_original[good_features].div(totals, axis=0).fillna(0)
        
        df_result = df_original.copy()
        df_result["attributed_dynamic_Ws"] = ratios.mul(df_budgets, axis=0).sum(axis=1).values
        

        plotter = AttributionPlotter(df_result.reset_index(), time_col="_time", energy_col="attributed_dynamic_Ws")
        plotter.plot_top_processes(top_n=8, save_path=f"{custom_name}_ebm_process.png")
        plotter.plot_top_processes_new(top_n=8, save_path=f"{custom_name}_ebm_process_new.png")
        plotter.plot_top_pids(top_n=8, save_path=f"{custom_name}_ebm_pid.png")
        
        return df_result