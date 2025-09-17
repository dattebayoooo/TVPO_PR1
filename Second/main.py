import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.array([5, 15, 25, 35, 45, 55], dtype=float).reshape(-1, 1)
y = np.array([5, 20, 14, 32, 22, 38], dtype=float)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("=== Simple Linear Regression (Conda env) ===")
print(f"coef_: {model.coef_[0]:.6f}")
print(f"intercept_: {model.intercept_:.6f}")
print(f"R^2: {r2_score(y, y_pred):.4f}")
