import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Define the input data
x = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
y = np.array([2, 8, 14, 28, 36, 52])

# Define the degree of the polynomial
degree = 3

# Create polynomial features
poly_features = PolynomialFeatures(degree=degree)
x_poly = poly_features.fit_transform(x)

# Train the model using linear regression
model = LinearRegression()
model.fit(x_poly, y)

# Make predictions
x_test = np.array([7, 8, 9, 10]).reshape((-1, 1))
x_test_poly = poly_features.transform(x_test)
y_pred = model.predict(x_test_poly)

# Plot the results
plt.scatter(x, y)
plt.plot(x_test, y_pred, color='red')
plt.show()
