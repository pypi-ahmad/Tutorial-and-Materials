# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
dataset = pd.read_csv('example_dataset.csv')

# Separate features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the decision tree regression model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Predict the test set results
y_pred = regressor.predict(X_test)

# Evaluate the model performance using R-squared metric
r2 = r2_score(y_test, y_pred)
print('R-squared score:', r2)

# In this code, we start by importing the necessary libraries, including pandas for data manipulation, sklearn for machine learning, and train_test_split and r2_score for model evaluation.

# We then load our example dataset and separate the features and target variable. We split the data into training and testing sets using the train_test_split function from sklearn.

# Next, we create an instance of the DecisionTreeRegressor class and fit it to the training data. We then use the trained model to make predictions on the test set using the predict method.

# Finally, we evaluate the performance of the model using the R-squared metric and print the result to the console.

# Note that this is just a basic example of how to implement decision tree regression in Python, and there are many options and hyperparameters that can be adjusted to improve model performance.



