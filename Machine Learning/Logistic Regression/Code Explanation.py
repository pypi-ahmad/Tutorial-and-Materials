# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("example_dataset.csv")

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df.drop("Target_Variable", axis=1), df["Target_Variable"], test_size=0.2)

# Create a Logistic Regression object
logreg = LogisticRegression()

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = logreg.predict(X_test)

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("Accuracy of the Logistic Regression model: {:.2f}%".format(accuracy*100))

# In this code, we first import the necessary libraries including pandas for loading and manipulating the dataset, scikit-learn for machine learning models, and NumPy for numerical calculations.

# We then load the example dataset and split it into training and testing data using train_test_split() function. The target variable is dropped from the training data.

# Next, we create an instance of the Logistic Regression class and fit it on the training data using fit() method. Then, we use predict() method to make predictions on the testing data.

# Finally, we calculate the accuracy of the model using accuracy_score() function from scikit-learn and print the result.

# Note that this is just a basic example to demonstrate the syntax of logistic regression in Python using scikit-learn. In real-world applications, data preprocessing and feature engineering steps may be necessary before training the model to achieve better results.