# test.py

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from KAN import KAN  # Убедитесь, что файл с моделью называется KAN.py

# ----------------- ШАГ 1: ПОДГОТОВКА ДАННЫХ -----------------
# Создаем обучающий набор данных
num_samples = 1000
X_train = torch.randn(num_samples, 2) # 1000 пар (x1, x2)

# Применяем нашу целевую функцию
# y = sin(pi * x1) + x2^2
y_train = torch.sin(np.pi * X_train[:, 0]) + X_train[:, 1]**2
y_train = y_train.unsqueeze(1) # Приводим к форме [1000, 1]

print(f"Форма обучающих данных X: {X_train.shape}")
print(f"Форма обучающих данных y: {y_train.shape}\n")


# ----------------- ШАГ 2: СОЗДАНИЕ МОДЕЛИ, ФУНКЦИИ ПОТЕРЬ И ОПТИМИЗАТОРА -----------------
# 1. Определяем архитектуру
kan_shape = [2, 5, 1] 

# 2. Создаем модель
model = KAN(shape=kan_shape, grid_size=5, spline_order=3)
print("Модель KAN успешно создана!")

# 3. Функция потерь (MSE)
loss_fn = nn.MSELoss()

# 4. Оптимизатор (Adam - хороший выбор для начала)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------- ШАГ 3: ЦИКЛ ОБУЧЕНИЯ -----------------
num_epochs = 1000
losses = []

print("Начинаем обучение...")
for epoch in range(num_epochs):
    # 1. Прямой проход: получаем предсказания модели
    predictions = model(X_train)
    
    # 2. Вычисление потерь: сравниваем предсказания с реальными данными
    loss = loss_fn(predictions, y_train)
    
    # 3. Обратное распространение ошибки
    optimizer.zero_grad() # Обнуляем градиенты с предыдущего шага
    loss.backward()       # Вычисляем градиенты для всех параметров
    optimizer.step()      # Обновляем параметры модели
    
    # Сохраняем и выводим информацию о потерях
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}/{num_epochs}, Потери (Loss): {loss.item():.6f}")

print("Обучение завершено!")

# ----------------- ШАГ 4: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТА -----------------
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("Динамика потерь (Loss) во время обучения")
plt.xlabel("Эпоха")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.yscale('log') # Логарифмическая шкала для лучшей наглядности
plt.show()

