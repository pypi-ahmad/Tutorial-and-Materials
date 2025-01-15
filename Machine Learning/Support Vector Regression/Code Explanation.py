from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

# Train the SVR model
regressor = SVR(kernel='linear')
regressor.fit(X_train, y_train)

# Predict the test set results
y_pred = sc_y.inverse_transform(regressor.predict(X_test))

# Print the predicted values
print("Predicted Values:")
print(y_pred)

# In this code example, we first import the necessary libraries - SVR from sklearn.svm, train_test_split from sklearn.model_selection, and StandardScaler from sklearn.preprocessing.

# Next, we create a sample dataset X and y to train and test the SVR model. We split the data into training and testing sets using train_test_split.

# After splitting the data, we apply feature scaling to both the training and testing sets using StandardScaler. This step is important because Support Vector Regression models are sensitive to the scale of the features.

# We then create an instance of the SVR model with a linear kernel and train it using the training set using the fit method.

# Finally, we use the predict method of the SVR model to predict the output for the test set, and then we use inverse_transform to transform the predicted values back to their original scale. The predicted values are then printed to the console.

# Note that this is a simple example to illustrate the syntax of Support Vector Regression using Scikit-Learn. In practice, you would want to perform more thorough model selection and tuning to obtain the best results for your specific problem.