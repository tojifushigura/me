# 13 Вариант Y = X1**6 + X2**2 + X1**3 + 4 * X2 + 5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Генерация данных
x1 = np.linspace(-10, 10, 400)  # 400 точек от -10 до 10
x2 = np.linspace(-10, 10, 400)  # 400 точек от -10 до 10

# Создаем сетку для x1 и x2
X1, X2 = np.meshgrid(x1, x2)

# Определяем функцию y
Y = X1**6 + X2**2 + X1**3 + 4 * X2 + 5

# Создание DataFrame и сохранение в CSV файл
data = pd.DataFrame({
    'x1': X1.ravel(),
    'x2': X2.ravel(),
    'y': Y.ravel()
})
data.to_csv('graph.csv', index=False)

# 2D график y(x1) при фиксированном x2
constant_x2 = 0
y_at_constant_x2 = constant_x2**2 + (x1**6 + x1**3 + 5 + 4 * constant_x2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x1, y_at_constant_x2, color='blue')
plt.title('График y(x1) при x2 = 0')
plt.xlabel('x1')
plt.ylabel('y')

# 2D график y(x2) при фиксированном x1
constant_x1 = 0
y_at_constant_x1 = (constant_x1**6 + constant_x1**3 + 5 + 4 * X2**2)

plt.subplot(1, 2, 2)
plt.plot(x2, y_at_constant_x1, color='red')
plt.title('График y(x2) при x1 = 0')
plt.xlabel('x2')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# Построение 3D графика
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)
ax.set_title('3D график y(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
