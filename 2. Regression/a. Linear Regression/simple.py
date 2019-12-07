# Importing the libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

##############################################################
# Preprocessing
cwd = os.getcwd()
dataset = pd.read_csv(cwd + '/salary.csv')

# Independent variables - every row except last column
X = dataset.iloc[:, :-1].values

# Dependent variables - only the last column
Y = dataset.iloc[:, 1].values

# Split the data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

###############################################################

# Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

# Visualise Training
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary LR (Training)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualise Test
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary LR (Test)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
