# %%
import os, sys
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.tools as stat
import statsmodels.api as sm
import statsmodels.formula.api as smf


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

# << Scikit-learn >>

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

# << Statsmodels >> 

# Load data
# test
dat = sm.datasets.get_rdataset("Guerry", "HistData").data
# Fit regression model
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

# %%
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# %% 
# Multi-variant linear regression -------------------------------

# << Scikit-learn >>

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

# %%

# << Statsmodels >> 

# X need to add constant to match the results from scikit learn
X1 = stat.add_constant(X)
model = sm.OLS(y, X1)
results = model.fit()
print(results.summary())

# %%
# Regression Model with Qualitative Predictors ---------------

# load data
path = '/Users/michaelshih/Documents/code/education/statistical_learining/'
subfolder = 'resource'
filename = 'Credit.csv'
filedir = os.path.join(path, subfolder, filename)
print(filedir)

data = pd.read_csv(filedir, index_col = 0)
data = pd.DataFrame(data)
display(data)

# %%
# make scatterplot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(data)
plt.show()

# %%
# seaborn
sns.pairplot(data)

# %%
X = data.loc[:, data.columns != 'Balance']
y = data['Balance']
# %%
X_encoded = pd.get_dummies(X)
X_encoded.head()

# %%
from sklearn.preprocessing import StandardScaler


# << Scikit-learn >>
# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_encoded, y)
score = model.score(X_encoded, y)
print(f"R2 Score: {score}")
print('Weight coefficients: ', model.coef_)

# %%
# << Statsmodels >> 
X1 = stat.add_constant(X_encoded)
model = sm.OLS(y, X1)
results = model.fit()
print(results.summary())