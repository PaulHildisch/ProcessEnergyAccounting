
import pandas as pd

class Preprocessor:

    def __init__(self, data ,good_features, target= "interval_energy"):
        #Check this if it takes too much memory
        self.df = data.copy()
        self.good_features = good_features
        self.target = target

    def _convert_datetime(self):
        self.df["_time"] = pd.to_datetime(self.df["_time"]).dt.round("1ms")

    def _fill_nan_values(self):

        for feature in self.good_features:
            if feature not in self.df.columns:
                self.df[feature] = 0.0

        self.df[self.good_features] = self.df[self.good_features].fillna(0)

    def _extract_interval_energy(self):
        interval_energy_all = (
            self.df[["_time", self.target]]
            .dropna()
            .drop_duplicates("_time")
            .set_index("_time")[self.target]
        )
        self.df = self.df[self.df["_time"].isin(interval_energy_all.index)]
        self.interval_energy_all = interval_energy_all.sort_index()

    def _aggregate(self):
        df_agg = self.df.groupby("_time")[self.good_features].sum()
        self.df_agg = df_agg.reindex(self.interval_energy_all.index).fillna(0)
    

    def preprocess(self):
        self._convert_datetime()
        self._fill_nan_values()
        self._extract_interval_energy()
        self._aggregate()
        X = self.df_agg.values
        y = self.interval_energy_all.values
        times = self.interval_energy_all.index.values
        return X, y, times
    


