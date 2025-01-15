from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=10, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an Elastic Net model with default hyperparameters
en = ElasticNet()

# Train the model on the training data
en.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = en.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)

print("Elastic Net model MSE:", mse)

# In this code, we first generate synthetic data using make_regression function from scikit-learn. Then, we split the data into training and testing sets using train_test_split function.

# Next, we create an instance of Elastic Net regression model using ElasticNet class. By default, the alpha parameter is set to 1.0 and l1_ratio is set to 0.5, which corresponds to a combination of L1 and L2 regularization.

# We then fit the model on the training data using the fit method, and use it to make predictions on the testing data using the predict method.

# Finally, we calculate the mean squared error of the predictions using the mean_squared_error function from scikit-learn's metrics module.

# Note that in practice, it's important to tune the hyperparameters of the Elastic Net model to achieve better performance on a specific task. This can be done using techniques such as cross-validation and grid search.