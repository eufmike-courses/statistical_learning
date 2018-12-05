# %%
import os, sys
import numpy as np
import pandas as pd
import timeit

# %% 
# load data
path = '/Users/michaelshih/Documents/code/education/statistical_learining/'
subfolder = 'resource'
filename = 'Advertising.csv'
filedir = os.path.join(path, subfolder, filename)
print(filedir)

data = pd.read_csv(filedir, index_col = 0)
data = pd.DataFrame(data)
display(data)
# %%
print(data.shape)
# %%
# Univarite linear regression -------------------------------
# select data for training
# the input 
X = np.array([data['TV']]).reshape(-1, 1)
y = np.array([data['sales']]).reshape(-1, 1)
print(X)


# %%
# performing linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
print(model)
print('Weight coefficients: ', model.coef_)
print('y-axis intercept: ', model.intercept_) 

# %%
# create predictions
predictions = model.predict(X)
print(f"True output: {y[0]}")
print(f"Predicted output: {predictions[0]}")
print(f"Prediction Error: {predictions[0]-y[0]}")

# %%
print(predictions)

# %%
report = pd.DataFrame({"Predicted": predictions[:, 0], "Actual": y[:, 0], "Error": (predictions - y)[:, 0]})
display(report)

# %%
from sklearn.metrics import mean_squared_error, r2_score
print("MSE: ", mean_squared_error(y, predictions))
print("Variance score: ", r2_score(y, predictions))

# %%
# plot dots
import matplotlib.pyplot as plt
plt.scatter(X, y, c='blue')

# plot regression line
x_min = min(X)
y_min = min(predictions)
x_max = max(X)
y_max = max(predictions)
plt.plot([x_min, x_max], [y_min, y_max], c='red')

# %%
# Multi-variant linear regression -------------------------------
X = np.array(data.loc[:, ['TV', 'radio', 'newspaper']])
print(X)
y = np.array([data['sales']]).reshape(-1, 1)

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
score = model.score(X, y)
print(f"R2 Score: {score}")
print('Weight coefficients: ', model.coef_)
