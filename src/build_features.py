import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np

X_train = pd.read_csv("data/processed/X_train.csv")
X_val = pd.read_csv("data/processed/X_val.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

categorical_cols = ['make_model', 'fuel', 'gear', 'offer_type', 'brand']
numerical_cols = ['mileage', 'power', 'age', 'mileage_per_year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(preprocessor, "models/preprocessor.pkl")


np.save("data/processed/X_train_processed.npy", X_train_processed)
np.save("data/processed/X_val_processed.npy", X_val_processed)
np.save("data/processed/X_test_processed.npy", X_test_processed)

print(f"Train shape: {X_train_processed.shape}")
