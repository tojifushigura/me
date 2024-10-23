#23 Вариант pip install ucimlrepo
#from ucimlrepo import fetch_ucirepo 
#  
## fetch dataset 
#spambase = fetch_ucirepo(id=94) 
#  
## data (as pandas dataframes) 
#X = spambase.data.features 
#y = spambase.data.targets 
#  
## metadata 
#print(spambase.metadata) 
#  
## variable information 
#print(spambase.variables) 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Загрузка набора данных Spambase
spambase = fetch_ucirepo(id=94)

# Данные
X = spambase.data.features
y = spambase.data.targets

# 1. Разделение данных на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 3. Обучение модели Perceptron
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron.fit(X_train_scaled, y_train)

# Предсказание и оценка точности Perceptron
y_pred_perceptron = perceptron.predict(X_test_scaled)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
print(f'Perceptron Accuracy: {accuracy_perceptron:.4f}')
print(classification_report(y_test, y_pred_perceptron))

# 4. Обучение модели MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Предсказание и оценка точности MLPClassifier
y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f'MLPClassifier Accuracy: {accuracy_mlp:.4f}')
print(classification_report(y_test, y_pred_mlp))

# 5. Эксперименты с гиперпараметрами для MLPClassifier
learning_rates = [0.001, 0.01, 0.1, 0.5]
alphas = [0.0001, 0.001, 0.01, 0.1]
train_accuracies = []
val_accuracies = []

for lr in learning_rates:
    for alpha in alphas:
        mlp = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=lr, alpha=alpha, max_iter=500, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        
        # Точность на обучающей и валидационной выборках
        train_accuracy = mlp.score(X_train_scaled, y_train)
        val_accuracy = mlp.score(X_val_scaled, y_val)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Learning rate: {lr}, Alpha: {alpha}')
        print(f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Построение графиков зависимости точности от гиперпараметров
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_accuracies, label='Training Accuracy', marker='o')
ax.plot(val_accuracies, label='Validation Accuracy', marker='o')
ax.set_title('Зависимость точности от гиперпараметров (MLPClassifier)')
ax.set_xlabel('Гиперпараметры (Learning rate, Alpha)')
ax.set_ylabel('Точность')
ax.legend()
plt.grid()
plt.show()
