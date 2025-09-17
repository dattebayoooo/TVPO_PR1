import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка датасета
diabetes = datasets.load_diabetes()

# Используем только одну фичу
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Разделяем данные на тренировочные и тестовые
X_train = diabetes_X[:-20]
X_test = diabetes_X[-20:]
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

# Создаем модель линейной регрессии
model = linear_model.LinearRegression()

# Обучаем модель
model.fit(X_train, y_train)

# Делаем предсказания
y_pred = model.predict(X_test)

# Выводим метрики
print("Коэффициенты: ", model.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Коэффициент детерминации: %.2f" % r2_score(y_test, y_pred))

# Визуализация
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
plt.title("Линейная регрессия - тест виртуального окружения")
plt.show()