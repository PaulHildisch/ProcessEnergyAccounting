
import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor:

    def __init__(self, data ,good_features, target= "interval_energy"):
        #Check this if it takes too much memory
        self.df = data.copy()
        self.good_features = good_features
        self.target = target

    def _convert_datetime(self):
        #Check this
        #self.df["_time"] = pd.to_datetime(self.df["_time"]).dt.round("1ms")
        self.df["_time"] = pd.to_datetime(self.df["_time"]).dt.round("1s")


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


    def _split(self):
        self.X_train, self.X_test, self.y_train, self.y_test, self.t_train, self.t_test = train_test_split(self.df_agg, self.interval_energy_all, self.interval_energy_all.index, test_size=0.2, shuffle=False)
    
    #TODO build different methods for this
    def _split_time_blocks(self):
        block_ids = self.df_agg.index.floor("10min").factorize()[0]
        is_test = (block_ids % 5 == 4)
        
        self.X_train = self.df_agg[~is_test]
        self.X_test  = self.df_agg[is_test]
        
        self.y_train = self.interval_energy_all[~is_test]
        self.y_test  = self.interval_energy_all[is_test]
        
        self.t_train = self.interval_energy_all.index[~is_test]
        self.t_test  = self.interval_energy_all.index[is_test]

    def preprocess(self, verbose=True ):
        if verbose:
            print("Converting to datetime")
        self._convert_datetime()
        if verbose:
            print("fill nan")
        self._fill_nan_values()
        if verbose:
            print("fill extract interval energy")
        self._extract_interval_energy()
        if verbose:
            print("aggregate")
        self._aggregate()
        self._split()
        if verbose:
            print("split")

        return self.X_train, self.X_test, self.y_train, self.y_test, self.t_train, self.t_test


    def preprocess_no_split(self):
        self._convert_datetime()
        print("Converting to datetime")
        self._fill_nan_values()
        print("fill nan")
        self._extract_interval_energy()
        print("fill extract interval energy")
        self._aggregate()
        print("aggregate")
        X = self.df_agg
        y = self.interval_energy_all
        t = self.interval_energy_all.index
        return X, y, t

    


