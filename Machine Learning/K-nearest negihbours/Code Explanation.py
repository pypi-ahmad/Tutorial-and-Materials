from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create a K-NN classifier object
knn = KNeighborsClassifier(n_neighbors=3)

# fit the classifier to the training data
knn.fit(X_train, y_train)

# predict the classes of new data points
y_pred = knn.predict(X_test)

# evaluate the performance of the classifier
accuracy = knn.score(X_test, y_test)

# print the accuracy of the classifier
print("Accuracy:", accuracy)

# In this code, we first load the iris dataset and split it into training and testing sets. We then create a KNeighborsClassifier object with n_neighbors=3, which means that the algorithm will consider the three nearest neighbors to a new data point when making a prediction.

# We fit the classifier to the training data using the fit() method, and then predict the classes of the test data using the predict() method. Finally, we evaluate the performance of the classifier using the score() method, which computes the accuracy of the classifier on the test data.

# The syntax of the K-NN algorithm in scikit-learn is quite simple and easy to understand. The KNeighborsClassifier class provides a number of parameters that can be adjusted to fine-tune the algorithm's performance, such as the number of neighbors to consider (n_neighbors) and the distance metric to use (metric). Overall, the scikit-learn implementation of K-NN is a powerful and flexible tool for classification tasks.