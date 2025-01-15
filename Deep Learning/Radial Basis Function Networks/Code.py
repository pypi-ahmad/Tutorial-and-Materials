import numpy as np

class RBFNetwork:
    
    def __init__(self, n_centers, learning_rate):
        self.n_centers = n_centers
        self.learning_rate = learning_rate
        self.centers = None
        self.widths = None
        self.weights = None
        
    def fit(self, X, y):
        # Randomly initialize centers
        self.centers = X[np.random.choice(X.shape[0], self.n_centers), :]
        
        # Calculate widths based on median distance between centers
        pairwise_dists = np.sqrt(np.sum((self.centers[:, np.newaxis] - X)**2, axis=2))
        self.widths = np.median(pairwise_dists) / np.sqrt(2 * self.n_centers)
        
        # Calculate activations for each example
        activations = np.exp(-(pairwise_dists ** 2) / (2 * self.widths ** 2))
        
        # Add bias term to activations
        activations = np.hstack([np.ones((X.shape[0], 1)), activations])
        
        # Calculate weights using pseudoinverse
        self.weights = np.linalg.pinv(activations) @ y
        
    def predict(self, X):
        pairwise_dists = np.sqrt(np.sum((self.centers[:, np.newaxis] - X)**2, axis=2))
        activations = np.exp(-(pairwise_dists ** 2) / (2 * self.widths ** 2))
        activations = np.hstack([np.ones((X.shape[0], 1)), activations])
        predictions = activations @ self.weights
        return predictions

# In this code, we define a class RBFNetwork that represents the Radial Basis Function network. The constructor takes two arguments: n_centers, which is the number of centers (or neurons) in the hidden layer, and learning_rate, which is the learning rate used in training the network.

# The fit method takes a set of training examples X and their corresponding labels y. It randomly initializes the centers, calculates the widths based on the median distance between centers, and calculates the activations for each example using the RBF activation function. It then adds a bias term to the activations and calculates the weights using the pseudoinverse.

# The predict method takes a set of test examples X and returns the predicted labels using the learned weights.

# Overall, the Radial Basis Function network is a powerful tool for function approximation and has many practical applications in fields such as finance, engineering, and image processing.