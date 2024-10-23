import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Генерация данных
x1 = np.linspace(-10, 10, 400)  # 400 px от -10 до 10
x2 = np.linspace(-10, 10, 400)  # 400 px от -10 до 10

# Создаем сетку для x1 и x2
X1, X2 = np.meshgrid(x1, x2)

# Вычисляем y = cos(x1) * x2^3
Y = np.cos(X1) * X2**3

# Сохраняем данные в CSV файл
data = pd.DataFrame({
    'x1': X1.ravel(),
    'x2': X2.ravel(),
    'y': Y.ravel()
})
data.to_csv('graph.csv', index=False)

# 2. Построение графиков
# График y(x1) при фиксированном x2 (например, x2 = 0)
constant_x2 = 0
y_at_constant_x2 = np.cos(x1) * constant_x2**3

plt.figure(figsize=(12, 6))

# График для y в зависимости от x1
plt.subplot(1, 2, 1)
plt.scatter(x1, y_at_constant_x2, color='blue', s=5)
plt.title('График y(x1) при x2 = 0')
plt.xlabel('x1')
plt.ylabel('y')

# График y(x2) при фиксированном x1 (например, x1 = 0)
constant_x1 = 0
y_at_constant_x1 = np.cos(constant_x1) * x2**3

plt.subplot(1, 2, 2)
plt.scatter(x2, y_at_constant_x1, color='red', s=5)
plt.title('График y(x2) при x1 = 0')
plt.xlabel('x2')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# 3. Вывод статистики
mean_values = data.mean()
min_values = data.min()
max_values = data.max()

print("Средние значения:\n", mean_values)
print("\nМинимальные значения:\n", min_values)
print("\nМаксимальные значения:\n", max_values)

# 4. Фильтрация данных
mean_x1 = data['x1'].mean()
mean_x2 = data['x2'].mean()
filtered_data = data[(data['x1'] < mean_x1) | (data['x2'] < mean_x2)]
filtered_data.to_csv('filtered_data.csv', index=False)

# 5. Построение 3D графика
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)
ax.set_title('3D график y(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
