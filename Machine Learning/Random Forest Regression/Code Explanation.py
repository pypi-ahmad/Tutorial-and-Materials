# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Loading the dataset
dataset = pd.read_csv('housing.csv')

# Preprocessing the data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating a Random Forest Regression model
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# Fitting the model to the training data
regressor.fit(X_train, y_train)

# Predicting the output for the test data
y_pred = regressor.predict(X_test)

# Evaluating the model's performance
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, y_pred)
print("R2 Score:", r2_score)
