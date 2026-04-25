import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Загружаем правильный датасет
df = pd.read_csv("data/raw/europe-motorbikes-zenrows.csv")

print(f"Начальный размер: {df.shape}")
print(f"Пропуски:\n{df.isnull().sum()}")

# --- 1. ОЧИСТКА ДАННЫХ ---
df = df.drop_duplicates()
df = df.dropna(subset=['price'])

# Обработка пропусков в power (заполняем медианой)
power_median = df['power'].median()
df['power'] = df['power'].fillna(power_median)

# Фильтрация выбросов по цене (отсекаем заглушки и нереальные цены)
df = df[(df['price'] > 100) & (df['price'] < 100000)]

# Фильтрация выбросов по пробегу (отсекаем скрученный/фейковый пробег)
df = df[(df['mileage'] >= 0) & (df['mileage'] < 300000)]

# Фильтрация выбросов по мощности (отсекаем нереальную мощность)
df = df[(df['power'] > 0) & (df['power'] < 500)]

print(f"После очистки: {df.shape}")

# --- 2. FEATURE ENGINEERING ---
# 1. Возраст мотоцикла (age)
# date обычно имеет формат "MM/YYYY", берем последние 4 символа как год
df['year'] = pd.to_numeric(df['date'].astype(str).str[-4:], errors='coerce')
df['year'] = df['year'].fillna(df['year'].median()) # заполняем пропуски, если дата была кривой
df['age'] = 2021 - df['year'] # Данные собраны в 2021 году
df['age'] = df['age'].apply(lambda x: x if x > 0 else 1) # чтобы не было деления на 0 для новых байков

# 2. Пробег в год (mileage_per_year)
df['mileage_per_year'] = df['mileage'] / df['age']

# 3. Марка (brand)
# Берем первое слово из make_model (например, "BMW" из "BMW F 800 GT")
df['brand'] = df['make_model'].astype(str).apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'Unknown')

# Удаляем колонки, которые больше не нужны или слишком грязные
df = df.drop(columns=['date', 'year', 'link', 'version'], errors='ignore')

print(f"После Feature Engineering: {df.shape}")

# --- 3. РАЗБИЕНИЕ ДАННЫХ ---
X = df.drop(columns=['price'])
y = df['price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_val.to_csv("data/processed/X_val.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
