# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create a Naive Bayes classifier object
gnb = GaussianNB()

# Train the classifier on the training data
gnb.fit(X_train, y_train)

# Use the trained classifier to make predictions on the testing data
y_pred = gnb.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("Accuracy: {:.2f}%".format(accuracy * 100))

# In this code example, we first import the necessary libraries including the iris dataset from sklearn.datasets, train_test_split from sklearn.model_selection, GaussianNB from sklearn.naive_bayes, and accuracy_score from sklearn.metrics.

# Then, we load the iris dataset and split it into training and testing sets using the train_test_split function. We allocate 70% of the data for training and the remaining 30% for testing. We set a random state of 42 to ensure reproducibility.

# Next, we create a Naive Bayes classifier object using the GaussianNB function from sklearn.naive_bayes.

# We then train the classifier on the training data using the fit method.

# After training, we use the trained classifier to make predictions on the testing data using the predict method.

# Finally, we calculate the accuracy of the classifier by comparing the predicted labels to the true labels using the accuracy_score method from sklearn.metrics.

# The accuracy of the classifier is then printed on the console in percentage format.

# Note that this is a simple example to demonstrate the use of the Naive Bayes algorithm. In a real-world scenario, you may need to perform additional steps such as data cleaning, feature engineering, and hyperparameter tuning to achieve better results.
