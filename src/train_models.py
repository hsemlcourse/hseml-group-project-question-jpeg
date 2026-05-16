import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time


def evaluate_model(model, X_train, y_train, X_val, y_val, param_grid, name):
    print(f"--- Обучение {name} ---")
    start_time = time.time()

    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"Лучшие параметры: {grid.best_params_}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    print(f"Время: {time.time() - start_time:.2f} сек\n")

    return {
        'model': name,
        'best_params': str(grid.best_params_),
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }, best_model


def main():
    # Загрузка данных
    X_train = np.load("data/processed/X_train_processed.npy")
    X_val = np.load("data/processed/X_val_processed.npy")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    # Фиксация seed
    SEED = 42

    models = {
        'Ridge': {
            'model': Ridge(random_state=SEED),
            'params': {'alpha': [0.1, 1.0, 10.0]}
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=SEED, n_jobs=-1),
            'params': {'n_estimators': [50], 'max_depth': [10, 20]}
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=SEED, objective='reg:squarederror', n_jobs=-1),
            'params': {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [5, 7, 9]}
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=-1),
            'params': {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63, 127]}
        },
        'CatBoost': {
            'model': CatBoostRegressor(random_state=SEED, verbose=0, thread_count=-1),
            'params': {'iterations': [100, 300], 'learning_rate': [0.05, 0.1], 'depth': [6, 8]}
        }
    }

    results = []
    best_overall_model = None
    best_mae = float('inf')
    best_model_name = ""

    for name, config in models.items():
        res, trained_model = evaluate_model(
            config['model'], X_train, y_train, X_val, y_val, config['params'], name
        )
        results.append(res)

        if res['MAE'] < best_mae:
            best_mae = res['MAE']
            best_overall_model = trained_model
            best_model_name = name

    # Сохранение результатов
    results_df = pd.DataFrame(results)

    # Добавляем baseline для сравнения
    if pd.io.common.file_exists("report/baseline_results.csv"):
        baseline_df = pd.read_csv("report/baseline_results.csv")
        baseline_df['best_params'] = "default"
        results_df = pd.concat([baseline_df, results_df], ignore_index=True)

    results_df.to_csv("report/experiments_results.csv", index=False)
    print("\nТаблица экспериментов сохранена в report/experiments_results.csv")

    # Сохраняем лучшую модель
    joblib.dump(best_overall_model, "models/final_model.pkl")
    print(f"Лучшая модель ({best_model_name}) сохранена в models/final_model.pkl")


if __name__ == "__main__":
    main()
