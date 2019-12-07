# Importing the libraries
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

##############################################################
# Importing the dataset
cwd = os.getcwd()
dataset = pd.read_csv(cwd + '/data.csv')

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

# Encoding categorical data - dependent variables
Y = LabelEncoder().fit_transform(Y)

###############################################################

# Split the data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

###############################################################

# Scaling the dataset

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

###############################################################
