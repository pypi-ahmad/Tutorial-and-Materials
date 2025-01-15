from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
X = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]]
y = [0, 1, 1, 2, 2, 3, 3, 4]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Ridge regression model with alpha=0.5
ridge_reg = Ridge(alpha=0.5)

# Train the model using the training set
ridge_reg.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = ridge_reg.predict(X_test)

# Evaluate the performance of the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)
