import joblib
import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import Ridge

class ModelBuilder():

    def __init__(self, X_train, X_test, y_train, y_test, model, scaler):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.scaler = scaler



    def _scale(self):
        #Try without scaling
        # self.scaler = None
        # self.X_train_scaled = self.X_train
        # self.X_test_scaled = self.X_test
        #self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train.values)
        self.X_test_scaled = self.scaler.transform(self.X_test.values)


    def _train(self):
        #self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        #self.model = Ridge(alpha=1.0)
        self.model.fit(self.X_train_scaled, self.y_train)
    
    def _test(self):
        self.y_pred = self.model.predict(self.X_test_scaled)
    
    def _evaluate(self):
        r2 = r2_score(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mean_energy = self.y_test.mean()
        mae_pct = (mae / mean_energy) * 100
        
        #print(f" Random Forest")
        print(f"  R² Score:  {r2:.4f}")
        print(f"  MAE:       {mae:.2f} Wh ({mae_pct:.2f}% of mean)")
        print("-" * 34)

    
    def _idle_power(self):
        #Predict an interval were all metrics are 0 to get an "idle prediction"
        zero_activity_interval = np.zeros((1, len(self.X_test_scaled[0])))
        zero_activity_interval = self.scaler.transform(zero_activity_interval)
        self.learned_idle_power = self.model.predict(zero_activity_interval)[0]
        print(f"The model's learned baseline idle interval energy is: {self.learned_idle_power:.2f} Wh")
        print("-" * 34)
        print("/n")

    def _save_model(self,path, filename):
        filepath = os.path.join(path, filename)
        bundle = {
            "scaler": self.scaler,
            "model": self.model
        }        
        joblib.dump(bundle, filepath)
        print(f"Model and scaler successfully saved to: {filepath}")
    
    def run_model(self,):
        self._scale()
        self._train()
        self._test()
        self._idle_power()
        self._evaluate()
        return self.y_pred, self.learned_idle_power
        
    
    def run_and_save_model(self, path="./", model_name="random_forest.joblib"):
        self._scale()
        self._train()
        self._test()
        self._evaluate()
        self._idle_power()
        #Dont svae rn
        #self._save_model(path, model_name)
        return self.y_pred, self.learned_idle_power

    