# Importing the libraries
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

##############################################################
# Importing the dataset
cwd = os.getcwd()
dataset = pd.read_csv(cwd + '/startups.csv')

# Independent variables - every row except last column
X = dataset.iloc[:, :-1].values

# Dependent variables - only the last column
Y = dataset.iloc[:, 4].values

###############################################################

# Encoding categorical data - independent variables
ct = ColumnTransformer(
    [('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]  # avoid dummy trap
###############################################################

# Split the data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

###############################################################

# Multiple Regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

###############################################################

# # Optimizing as backward elimination

# adding the constant values for b0 of equation
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# 1. Select a Significance level (5 percent here)
# 2. Taking all the columns first (predictors) and fit model
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

# 3. Get the p-values
print(regressor_OLS.summary())

# 4. Remove the highest p-value and fit model again (repeat till highest P-value is less than SL)
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())
