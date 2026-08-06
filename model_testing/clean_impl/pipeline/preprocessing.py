import pandas as pd
from sklearn.model_selection import train_test_split

#Preprocessing class to prepare collected data for model training


class Preprocessor:

    def __init__(self, data ,good_features, target= "interval_energy"):
        #Check this if it takes too much memory
        self.df = data.copy()
        self.good_features = good_features
        self.target = target

    def _convert_datetime(self):
        self.df["_time"] = pd.to_datetime(self.df["_time"]).dt.round("1ms")
        #self.df["_time"] = pd.to_datetime(self.df["_time"]).dt.round("1s")
        #just to be sure
        self.df = self.df.sort_values("_time")


    def _save_unaggregated_data(self):
        #needed for the process attribution
        self.df_unaggregated = self.df.set_index("_time")


    def _fill_nan_values(self):
        #print("Fill up potential nan values")
        for feature in self.good_features:
            if feature not in self.df.columns:
                self.df[feature] = 0.0

        self.df[self.good_features] = self.df[self.good_features].fillna(0)

    def _extract_interval_energy(self):
        #print("Extracting interval energy")
        interval_energy_all = (
            self.df[["_time", self.target]]
            .dropna()
            .drop_duplicates("_time")
            .set_index("_time")[self.target]
        )
        self.df = self.df[self.df["_time"].isin(interval_energy_all.index)]
        self.interval_energy_all = interval_energy_all.sort_index()


    def _aggregate(self):
        #print("Aggregating data by intervals")
        df_agg = self.df.groupby("_time")[self.good_features].sum()
        self.df_agg = df_agg.reindex(self.interval_energy_all.index).fillna(0)

    #Remove extreme outliers, the max_deviation_energy needs to be manually selected for the given node, since this heavily depends on the overall nodes power consumption
    def _remove_outliers(self, window, max_deviation_energy):
        #print("Remove Outliers")
        rolling_median = self.interval_energy_all.rolling(window = window, center = True).median()
        rolling_median = rolling_median.fillna(self.interval_energy_all)
        deviation = (self.interval_energy_all - rolling_median).abs()
        valid = deviation <= max_deviation_energy
        valid_times = self.interval_energy_all[valid].index

        outliers_dropped = (~valid).sum()
        print(f"Dropped {outliers_dropped} timestamps.")

        self.interval_energy_all = self.interval_energy_all.loc[valid_times]
        self.df_agg = self.df_agg.loc[valid_times]        
        self.df = self.df[self.df["_time"].isin(valid_times)]        
        if hasattr(self, 'df_unaggregated'):
            self._save_unaggregated_data()


    def _split(self):
        print("Splitting data into train and test sets")
        self.X_train, self.X_test, self.y_train, self.y_test, self.t_train, self.t_test = train_test_split(self.df_agg, self.interval_energy_all, self.interval_energy_all.index, test_size=0.2, shuffle=False)
    

    #Deprecated
    #Creates a 80/20 split: Was not used anymore as the project progressed
    def preprocess(self):
        self._convert_datetime()
        self._fill_nan_values()
        self._extract_interval_energy()
        self._aggregate()
        # self._save_unaggregated_data()
        # self._remove_outliers(window=5, max_deviation_energy=  150)# adjust this to the node
        self._split()

        return self.X_train, self.X_test, self.y_train, self.y_test, self.t_train, self.t_test

    #use this method for training and predicting on total workflows
    #does not explicitly split the data
    def preprocess_no_split(self):
        self._convert_datetime()
        self._fill_nan_values()
        self._extract_interval_energy()
        self._aggregate()
        self._save_unaggregated_data()
        self._remove_outliers(window=5, max_deviation_energy=  150)# adjust this to the node

        X = self.df_agg
        y = self.interval_energy_all
        t = self.interval_energy_all.index
        unaggregated = self.df_unaggregated
        return X, y, t, unaggregated

    


