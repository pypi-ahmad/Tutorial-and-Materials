# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate random classification dataset
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2,
                            n_redundant=0, random_state=0, shuffle=False)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Random Forest classifier object
clf = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the classifier using the training dataset
clf.fit(X_train, y_train)

# Make predictions on the testing dataset
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("Accuracy:", accuracy)

# In this example code, we first generate a random classification dataset using make_classification function from sklearn.datasets module. We then split the dataset into training and testing sets using train_test_split function from sklearn.model_selection module.

# Next, we create a Random Forest classifier object using RandomForestClassifier class from sklearn.ensemble module. We set the number of trees in the forest to be 100 and the random state to be 0.

# We then train the classifier using the training dataset by calling the fit method of the classifier object.

# After training, we make predictions on the testing dataset using the predict method of the classifier object.

# Finally, we calculate the accuracy of the classifier by comparing the predicted labels with the true labels using the accuracy_score function from sklearn.metrics module. The accuracy score is printed to the console.

# Note that this is just a basic example code, and there are many other parameters that can be tuned and techniques that can be used to improve the performance of Random Forest classifier.



