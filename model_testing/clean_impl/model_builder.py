import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

class ModelBuilder():

    def __init__(self, X, y, times):
        self.X = X
        self.y = y
        self.times = times
    
    def _split(self):
        self.X_train, self.X_test, self.y_train, self.y_test, self.t_train, self.t_test = train_test_split(self.X, self.y, self.times, test_size=0.2, shuffle=False)

    def _scale(self):
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def _train(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train_scaled, self.y_train)
    
    def _test(self):
        self.y_pred = self.model.predict(self.X_test_scaled)
    
    def _evaluate(self):
        r2 = r2_score(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mean_energy = self.y_test.mean()
        mae_pct = (mae / mean_energy) * 100
        
        #print(f"Random Forest: [{self.X_test_scaled.columns}]")
        print(f" Random Forest")
        print(f"  R² Score:  {r2:.4f}")
        print(f"  MAE:       {mae:.2f} Wh ({mae_pct:.2f}% of mean)")
        print("-" * 34)

    def _save_model(self,path, filename):
        filepath = os.path.join(path, filename)
        bundle = {
            "scaler": self.scaler,
            "model": self.model
        }        
        joblib.dump(bundle, filepath)
        print(f"Model and scaler successfully saved to: {filepath}")
    
    def run_model(self,):
        self._split()
        self._scale()
        self._train()
        self._test()
        self._evaluate()
        return self.y_pred, self.y_test, self.t_test
        
    
    def run_and_save_model(self, path="./", model_name="random_forest.joblib"):
        self._split()
        self._scale()
        self._train()
        self._test()
        self._evaluate()
        self._save_model(path, model_name)
        return self.y_pred, self.y_test, self.t_test

    