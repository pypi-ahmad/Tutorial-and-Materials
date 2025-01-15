import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Loading the dataset
dataset = pd.read_csv('example_dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Evaluating the model
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, y_pred)
print('R-squared value:', r2_score)
