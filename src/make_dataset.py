import pandas as pd
from sklearn.model_selection import train_test_split

# Загружаем правильный датасет
df = pd.read_csv("data/raw/europe-motorbikes-zenrows.csv")

print(f"Начальный размер: {df.shape}")
print(f"Пропуски:\n{df.isnull().sum()}")

# --- 1. ОЧИСТКА ДАННЫХ ---
df = df.drop_duplicates()
df = df.dropna(subset=['price'])

# Фильтрация выбросов по цене (отсекаем заглушки и нереальные цены)
df = df[(df['price'] > 100) & (df['price'] < 100000)]

# Фильтрация выбросов по пробегу (отсекаем скрученный/фейковый пробег)
df = df[(df['mileage'] >= 0) & (df['mileage'] < 300000)]

# Фильтрация выбросов по мощности (отсекаем нереальную мощность)
# Важно: сохраняем NaN, чтобы заполнить их позже (предотвращение Data Leakage)
df = df[df['power'].isna() | ((df['power'] > 0) & (df['power'] < 500))]

# Извлекаем год до сплита, чтобы было проще, но НЕ заполняем пропуски
df['year'] = pd.to_numeric(df['date'].astype(str).str[-4:], errors='coerce')

print(f"После очистки: {df.shape}")

# --- 2. РАЗБИЕНИЕ ДАННЫХ ---
X = df.drop(columns=['price'])
y = df['price']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# --- 3. ИМПЬЮТАЦИЯ И FEATURE ENGINEERING ---
# Вычисляем статистики ТОЛЬКО по X_train для предотвращения Data Leakage
power_median = X_train['power'].median()
year_median = X_train['year'].median()


def engineer_features(X_df):
    X_df = X_df.copy()

    # Импьютация
    X_df['power'] = X_df['power'].fillna(power_median)
    X_df['year'] = X_df['year'].fillna(year_median)

    # 1. Возраст мотоцикла (age)
    X_df['age'] = 2021 - X_df['year']  # Данные собраны в 2021 году
    # чтобы не было деления на 0 для новых байков
    X_df['age'] = X_df['age'].apply(lambda x: x if x > 0 else 1)

    # 2. Пробег в год (mileage_per_year)
    X_df['mileage_per_year'] = X_df['mileage'] / X_df['age']

    # 3. Марка (brand)
    # Берем первое слово из make_model (например, "BMW" из "BMW F 800 GT")
    X_df['brand'] = X_df['make_model'].astype(str).apply(
        lambda x: x.split()[0] if len(x.split()) > 0 else 'Unknown'
    )

    # Удаляем колонки, которые больше не нужны или слишком грязные
    X_df = X_df.drop(
        columns=['date', 'year', 'link', 'version'],
        errors='ignore'
    )

    return X_df


X_train = engineer_features(X_train)
X_val = engineer_features(X_val)
X_test = engineer_features(X_test)

print(f"После Feature Engineering (Train): {X_train.shape}")

# Сохранение
X_train.to_csv("data/processed/X_train.csv", index=False)
X_val.to_csv("data/processed/X_val.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
