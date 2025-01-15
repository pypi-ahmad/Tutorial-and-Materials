from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for scaling the features and running SGDRegressor
model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))

# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)

# Print the score
print("R-squared score:", score)

# In this code, we first load the Boston Housing dataset and split it into training and testing sets using the train_test_split() function. We then create a pipeline using make_pipeline() that scales the features using StandardScaler() and runs SGDRegressor() with a maximum of 1000 iterations and a tolerance level of 1e-3.

# We then fit the model on the training data using fit() and evaluate it on the testing data using score(), which calculates the R-squared score. Finally, we print the score using print().