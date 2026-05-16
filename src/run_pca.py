import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def main():
    print("Запуск PCA...")
    # Загрузка данных
    X_train = np.load("data/processed/X_train_processed.npy")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    # Применяем PCA, оставляем 2 компоненты для визуализации
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)

    print(f"Доля объясненной дисперсии (2 компоненты): {sum(pca.explained_variance_ratio_):.4f}")

    # Создаем DataFrame для удобства визуализации
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca['price'] = y_train

    # Визуализация
    plt.figure(figsize=(10, 8))
    # Для лучшей визуализации ограничим цены (например, до 95-го перцентиля), чтобы выбросы не портили цветовую шкалу
    price_cap = np.percentile(y_train, 95)

    scatter = plt.scatter(
        df_pca['PC1'],
        df_pca['PC2'],
        c=np.clip(df_pca['price'], 0, price_cap),
        cmap='viridis',
        alpha=0.5,
        s=10
    )
    plt.colorbar(scatter, label='Price (capped at 95th percentile)')
    plt.title('PCA Projection of Motorcycle Data (2 Components)')
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1] * 100:.1f}%)')

    os.makedirs("report/images", exist_ok=True)
    plt.savefig("report/images/pca_projection.png")
    print("График PCA сохранен в report/images/pca_projection.png")


if __name__ == "__main__":
    main()
