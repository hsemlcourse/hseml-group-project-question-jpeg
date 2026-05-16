FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости для сборки (если потребуются) и libomp для LightGBM/XGBoost (macOS/Linux compat)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Команда по умолчанию: запускаем пайплайн целиком
CMD ["sh", "-c", "python src/make_dataset.py && python src/build_features.py && python src/train_baseline.py && python src/run_pca.py && python src/train_models.py"]