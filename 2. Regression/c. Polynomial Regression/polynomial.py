# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

##############################################################
# Importing the dataset
cwd = os.getcwd()
dataset = pd.read_csv(cwd + '/pos_sal.csv')

# Independent variables - every row except last column
X = dataset.iloc[:, 1:2].values

# Dependent variables - only the last column
Y = dataset.iloc[:, 2].values

###############################################################

# Model and Predict
linreg = LinearRegression()
linreg.fit(X, Y)

polyreg = PolynomialFeatures(degree=5)
X_poly = polyreg.fit_transform(X)
linreg2 = LinearRegression()
linreg2.fit(X_poly, Y)
z = linreg2.predict(polyreg.fit_transform(X))

# Visualise
plt.scatter(X, Y, color='red')
plt.plot(X, linreg.predict(X), color='blue')
plt.title('Linear Reg - Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, linreg2.predict(polyreg.fit_transform(X_grid)), color='blue')
plt.title('Poly Reg - Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

##############################################################

# Final comparison
pred1 = linreg.predict([[6.5]])
pred2 = linreg2.predict(polyreg.fit_transform([[6.5]]))
