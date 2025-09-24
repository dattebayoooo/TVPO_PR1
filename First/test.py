import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# Загружаем датасет диабета
X, y = datasets.load_diabetes(return_X_y=True)

# Берём одну фичу
X = X[:, np.newaxis, 2]

# Делим на train/test
X_train, X_test = X[:-20], X[-20:]
y_train, y_test = y[:-20], y[-20:]

# Линейная регрессия
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print("Коэффициенты:", regr.coef_)
print("Среднеквадратичная ошибка:", np.mean((y_pred - y_test) ** 2))

plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.show()
