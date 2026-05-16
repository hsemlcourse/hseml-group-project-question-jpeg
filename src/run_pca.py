import numpy as np
from sklearn.decomposition import PCA


def main():
    print("Запуск PCA...")
    # Загрузка данных
    X_train = np.load("data/processed/X_train_processed.npy")

    # Применяем PCA, оставляем 2 компоненты
    pca = PCA(n_components=2, random_state=42)
    pca.fit_transform(X_train)

    print(f"Доля объясненной дисперсии (2 компоненты): {sum(pca.explained_variance_ratio_):.4f}")


if __name__ == "__main__":
    main()
