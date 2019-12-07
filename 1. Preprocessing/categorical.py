# Importing the libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################################
# Importing the dataset
cwd = os.getcwd() + "/1. Preprocessing/"
dataset = pd.read_csv(cwd + 'data.csv')

# Independent variables - every row except last column
X = dataset.iloc[:, :-1].values

# Dependent variables - only the last column
Y = dataset.iloc[:, 3].values

###############################################################

# Missing Data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

###############################################################

# Encoding categorical data - independent variables
ct = ColumnTransformer(
    [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
print(X)

# Encoding categorical data - dependent variables
Y = LabelEncoder().fit_transform(Y)
print(Y)

###############################################################
