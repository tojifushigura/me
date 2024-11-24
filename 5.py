# 11 Вариант 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from ucimlrepo import fetch_ucirepo

# Загрузка набора данных
user_knowledge_modeling = fetch_ucirepo(id=257)

# Извлечение данных в виде pandas DataFrame
X = user_knowledge_modeling.data.features
y = user_knowledge_modeling.data.targets

# Вывод метаданных
print("Метаданные набора данных:")
print(user_knowledge_modeling.metadata)
print("\nИнформация о переменных:")
print(user_knowledge_modeling.variables)

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Инициализация алгоритмов кластеризации
algorithms = {
    'KMeans': KMeans(n_clusters=3, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=3)
}

# Словарь для хранения результатов кластеризации
results = {}

# Кластеризация и визуализация результатов
for name, algorithm in algorithms.items():
    # Применение алгоритма кластеризации
    labels = algorithm.fit_predict(X_scaled)
    results[name] = labels

    # Визуализация результатов (используем только первые два признака для графика)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title(f'Результаты кластеризации с помощью {name}')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.colorbar(label='Класс')
    plt.show()

# Сравнение результатов кластеризации
for name, labels in results.items():
    print(f"\nРезультаты кластеризации с помощью {name}:")
    unique_labels = np.unique(labels)
    print(f"Количество уникальных классов: {len(unique_labels)}")
