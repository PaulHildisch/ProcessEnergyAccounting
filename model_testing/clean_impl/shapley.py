import shap
import numpy as np
import pandas as pd
from attributor_plotter import AttributionPlotter

class ProcessAttributor:
    # It would be better to use unscaled values, -> increases explainability
    #X_test was scaled for training also pass scaled here, otherwise not
    def __init__(self, model, X_test, idle_power):
        self.model = model
        self.X_test = X_test
        self.idle_power = idle_power
    

    def _init_explainer(self):
        self.explainer = shap.TreeExplainer(self.model)
        #should this be X_train?
        shap_values = self.explainer.shap_values(self.X_test)
        #This should be idle power
        #base_power = self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
        #print("Wh", base_power)

    def explain(self):
        self._init_explainer()

    def attribute(self,preds, X, good_features, t_test):
        self._init_explainer()
        df = self.attribute_entire_dataset(preds, self.explainer, X, good_features, t_test, self.idle_power)
        # top_consumers = df.groupby('process_name')['attributed_dynamic_Wh'].sum().sort_values(ascending=False)
        # print(top_consumers.head(20))
        # print("influx meand")
        # influx_sum = df.loc[df['process_name'] == "influxd", 'attributed_dynamic_Wh'].sum()
        
        # # Count the total number of unique timestamps in the ENTIRE test
        # total_intervals = df.index.nunique()
        
        # # Calculate the true global average
        # influx_global_mean = influx_sum / total_intervals
        
        # print(f"Total Sum: {influx_sum}")
        # print(f"Global Average: {influx_global_mean}")
        # (Your existing console prints...)
        top_consumers = df.groupby('process_name')['attributed_dynamic_Wh'].sum().sort_values(ascending=False)
        print(top_consumers.head(20))
        
        # 2. Trigger the plotter!
        plotter = AttributionPlotter(df, time_col="_time", energy_col="attributed_dynamic_Wh")
        
        # Plot Top Base Processes (grouped)
        plotter.plot_top_processes(top_n=8, save_path="shap_process_attribution.png")
        
        # Plot Top Individual PIDs
        plotter.plot_top_pids(top_n=50, save_path="shap_pid_attribution.png")

    def attribute_entire_dataset(self, preds, explainer, df_original, features, test_times, idle_power=305.0):
    # 1. Vectorized Power & Budgets (NumPy)
        #preds = model.predict(X_test_scaled)
        p_dynamic = np.maximum(0, preds - idle_power)
        
        # Calculate absolute SHAP and feature weights as matrices
        shap_vals = np.abs(explainer.shap_values(self.X_test))
        total_shap = np.sum(shap_vals, axis=1, keepdims=True)
        total_shap[total_shap == 0] = 1 # Prevent division by zero
        
        # Feature budgets: shape (n_intervals, n_features)
        feature_budgets = (shap_vals / total_shap) * p_dynamic[:, np.newaxis]
        #this contains how much dymanic power each feature contributes at a given time frame
        df_budgets = pd.DataFrame(feature_budgets, columns=features, index=test_times)
        print(df_budgets)
        df_budgets.index = pd.to_datetime(df_budgets.index)
        if df_budgets.index.tz is None and df_original.index.tz is not None:
            df_budgets.index = df_budgets.index.tz_localize(df_original.index.tz)
        
        # 2. Calculate totals per interval for denominator
        interval_totals = df_original.groupby("_time")[features].sum()
        
        # 3. Vectorized Distribution to Processes (Pandas)
        df_result = df_original.copy()
        df_result["attributed_dynamic_Wh"] = 0.0
        
        for feat in features:
            # Fast lookup mapping timestamp to interval total and budget
            total_usage = df_result.index.map(interval_totals[feat])
            budget = df_result.index.map(df_budgets[feat])
            
            # Calculate share: (process_usage / total_usage) * budget
            feat_share = np.where(total_usage > 0, (df_result[feat] / total_usage) * budget, 0)
            df_result["attributed_dynamic_Wh"] += feat_share
            
        return df_result


# CHECK IF THE ATTRIBUTION FOR TMUX STAYS the same across many different bench marks
# adjust for time

# build plotter
# try to verify this somehow



