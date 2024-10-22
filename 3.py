#13 Вариант 
# pip install ucimlrepo

# from ucimlrepo import fetch_ucirepo 
  
# # fetch dataset 
# energy_efficiency = fetch_ucirepo(id=242) 
  
# # data (as pandas dataframes) 
# X = energy_efficiency.data.features 
# y = energy_efficiency.data.targets 
  
# # metadata 
# print(energy_efficiency.metadata) 
  
# # variable information 
# print(energy_efficiency.variables) 
#####################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка набора данных
energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# 1. Разделение данных на обучающую и тестовую выборки
np.random.seed(42)  # Для воспроизводимости
indices = np.random.permutation(len(X))
train_size = int(len(X) * 0.8)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

# 2. Обучение модели линейной регрессии
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = lin_reg.predict(X_test)

# Проверка точности модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Линейная регрессия: MSE = {mse:.2f}, R² = {r2:.2f}")

# 3. Построение полиномиальной модели
degrees = [1, 2, 3, 4, 5]
train_scores = []
test_scores = []

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.transform(X_test)
    
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    
    train_pred = poly_reg.predict(X_poly_train)
    test_pred = poly_reg.predict(X_poly_test)
    
    train_scores.append(r2_score(y_train, train_pred))
    test_scores.append(r2_score(y_test, test_pred))

# Построение графиков зависимости точности от степени полинома
plt.figure(figsize=(10, 5))
plt.plot(degrees, train_scores, label='Train R²', marker='o')
plt.plot(degrees, test_scores, label='Test R²', marker='o')
plt.title('Точность полиномиальной регрессии в зависимости от степени полинома')
plt.xlabel('Степень полинома')
plt.ylabel('R²')
plt.legend()
plt.grid()
plt.show()

# 4. Модель с регуляризацией (Ridge Regression)
ridge_train_scores = []
ridge_test_scores = []
alphas = np.logspace(-4, 4, 10)

for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_train, y_train)
    
    ridge_train_pred = ridge_reg.predict(X_train)
    ridge_test_pred = ridge_reg.predict(X_test)
    
    ridge_train_scores.append(r2_score(y_train, ridge_train_pred))
    ridge_test_scores.append(r2_score(y_test, ridge_test_pred))

# Построение графиков зависимости точности от коэффициента регуляризации
plt.figure(figsize=(10, 5))
plt.plot(alphas, ridge_train_scores, label='Train R²', marker='o')
plt.plot(alphas, ridge_test_scores, label='Test R²', marker='o')
plt.xscale('log')
plt.title('Точность Ridge-регрессии в зависимости от коэффициента регуляризации')
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('R²')
plt.legend()
plt.grid()
plt.show()
