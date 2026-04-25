import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

X_train = np.load("data/processed/X_train_processed.npy")
X_val = np.load("data/processed/X_val_processed.npy")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"Baseline Linear Regression:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

results = pd.DataFrame([{'model': 'LinearRegression', 'MAE': mae, 'RMSE': rmse, 'R2': r2}])
results.to_csv("report/baseline_results.csv", index=False)

import joblib
joblib.dump(model, "models/baseline_model.pkl")