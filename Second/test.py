import numpy as np
from sklearn.linear_model import LinearRegression

# Данные
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Модель
model = LinearRegression()
model.fit(X, y)

print("Коэффициент наклона:", model.coef_[0])
print("Свободный член:", model.intercept_)

# Предсказание
print("Предсказание для 6:", model.predict([[6]])[0])
