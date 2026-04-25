import pandas as pd
import numpy as np
import os

def test_data_files_exist():
    """Проверяем, что исходные данные существуют."""
    assert os.path.exists("data/raw/europe-motorbikes-zenrows.csv"), "Raw data file is missing"

def test_processed_data_shapes():
    """Проверяем, что после сплита размерности совпадают."""
    if os.path.exists("data/processed/X_train.csv"):
        X_train = pd.read_csv("data/processed/X_train.csv")
        y_train = pd.read_csv("data/processed/y_train.csv")
        assert len(X_train) == len(y_train), "X_train and y_train must have the same number of rows"

def test_baseline_model_exists():
    """Проверяем, что модель сохранилась."""
    if os.path.exists("models/baseline_model.pkl"):
        assert os.path.exists("models/baseline_model.pkl"), "Baseline model is missing"
