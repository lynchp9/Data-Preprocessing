reset -f 
# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('')
#Separating Numerical Data
X_num = dataset.iloc[:,  ].values
#Separating Categorical Data
X_cat = dataset.iloc[:,  ].values
#Separating y Data
y = dataset.iloc[:,  ].values

# Feature Scaling (if Nessesary)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_num = sc_X.fit_transform(X_num)
y = sc_y.fit_transform(y)

# Fitting Polynomial Scaling to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_num = poly_reg.fit_transform(X_num)

# Encoding categorical data
#Use if X_cat is vector --> X_cat = X_cat.reshape(len(X_cat), 1)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X_cat[:, ] = labelencoder.fit_transform(X_cat[:, ])
onehotencoder = OneHotEncoder(categorical_features = [])
X_cat = onehotencoder.fit_transform(X_cat).toarray()

# Avoiding the Dummy Variable Trap
X_cat = X_cat[:, :-1]

#Adding categorical Data
X = np.c_[X_num, X_cat]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test, y_test, test_size = 0.5, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

reset -f