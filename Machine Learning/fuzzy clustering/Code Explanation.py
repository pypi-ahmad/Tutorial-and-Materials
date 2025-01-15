import numpy as np
import skfuzzy as fuzz

# Generate some data for clustering
np.random.seed(42)
data = np.vstack((np.random.randn(100, 2) * 0.5 + [2, 2],
                  np.random.randn(100, 2) * 0.5 + [-2, -2],
                  np.random.randn(100, 2) * 0.5 + [-2, 2]))

# Define the number of clusters
k = 3

# Fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, k, 2, error=0.005, maxiter=1000)

# Output the fuzzy partition matrix
print(u)

# Output the cluster centers
print(cntr)

# In this example, we first generate some two-dimensional data with three clusters using NumPy. Then, we define the number of clusters k and use the fuzz.cluster.cmeans() function from the scikit-fuzzy library to perform fuzzy c-means clustering on the data. The function takes the data as input, along with the number of clusters, a fuzziness parameter (m), an error tolerance, and a maximum number of iterations.

# The output of the fuzz.cluster.cmeans() function is a tuple containing the cluster centers (cntr) and the fuzzy partition matrix (u), which represents the degree of membership of each data point to each cluster.

# Finally, we print the fuzzy partition matrix and the cluster centers to the console. Note that the cntr array has shape (k, n_features), where n_features is the number of features in the data (in this case, 2).



