# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from importlib.resources import Resource

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Initial Data
x = np.array([8, 9, 8, 7, 9])
x1 = np.array([[8, 9, 8, 7, 9], [1, 1, 1, 0, 0]])
y = np.array([5, 9, 7, 3, 0])
plt.scatter(x, y)
x_reshaped = x.reshape(-1, 1)
x_transposed = np.transpose(x1)


# Util functions
def rmse(k: list):
    return sum((((k[0] * x + k[1]) - y) ** 2) / len(x))


def rmse_2(k: list):
    return sum((((k[0] * x1[0] + k[1] * x1[1] + k[2]) - y) ** 2) / len(x))


def line(x, k, b):
    return x * k + b


# Scipy minimize
result = minimize(rmse, np.array([0, 0]))
# print(result) k = 0.42, b = 1.28
x_guess = [line(i, 0.42, 1.28) for i in x]
# plt.plot(x, x_guess)

# Linear regression
linear_model = LinearRegression()
linear_model.fit(x_reshaped, y)
y_linear_predicted = linear_model.predict(x_reshaped)
# print(mean_squared_error(y, y_predicted))
# print(linear_model.coef_) k = 0.42
# print(linear_model.intercept_) b = 1.28
# plt.plot(x, y_linear_predicted)

# Ridge
ridge_model = Ridge(0.1)
ridge_model.fit(x_reshaped, y)
y_ridge_predicted = ridge_model.predict(x_reshaped)
# print(ridge_model.coef_) k = 0.41
# print(ridge_model.intercept_) k = 1.40
# plt.plot(x, y_ridge_predicted)

# Lasso
lasso_model = Lasso(0.1)
lasso_model.fit(x_reshaped, y)
y_lasso_predicted = lasso_model.predict(x_reshaped)
# print(lasso_model.coef_) k = 0.25
# print(lasso_model.intercept_) b = 2.75
# plt.plot(x, y_lasso_predicted)
# plt.show()

# Scipy minimize 2
result_2 = minimize(rmse_2, np.array([0, 0, 0]))
# print(result_2) k1 = -0.38, k2 = 5.63, b = 4.5

# Linear regression 2
linear_model_2 = LinearRegression()
linear_model_2.fit(x_transposed, y)
# print(linear_model_2.coef_) k1 = -0.38, k2 = 5.63
# print(linear_model_2.intercept_) b = 4.5
y_linear_predicted_2 = linear_model_2.predict(x_transposed)
# print(mean_squared_error(y, y_linear_predicted_2))

# Ridge 2
ridge_model_2 = Ridge(0.1)
ridge_model_2.fit(x_transposed, y)
# print(ridge_model_2.coef_) k1 = -0.3, k2 = 5.2
# print(ridge_model_2.intercept_) b = 4.2
y_ridge_predicted_2 = ridge_model_2.predict(x_transposed)
# print(mean_squared_error(y, y_ridge_predicted_2))

# Lasso 2
lasso_model_2 = Lasso(0.1)
lasso_model_2.fit(x_transposed, y)
# print(lasso_model_2.coef_) k1 = -0.12, k2 = 5.12
# print(lasso_model_2.intercept_) b = 2.74
y_lasso_predicted_2 = lasso_model_2.predict(x_transposed)
# print(mean_squared_error(y, y_lasso_predicted_2))
