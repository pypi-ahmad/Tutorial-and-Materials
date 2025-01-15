from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[5,3], [10,15], [15,12], [24,10], [30,45], [85,70], [71,80], [60,78], [55,52], [80,91]])

# Creating KMeans object for clustering with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fitting the data to KMeans object
kmeans.fit(X)

# Printing the labels and centroids
print("Labels: ", kmeans.labels_)
print("Centroids: ", kmeans.cluster_centers_)


# In this code, we first import the KMeans class from sklearn.cluster module. We then create a numpy array X with some sample data points for clustering.

# Next, we create a KMeans object with n_clusters parameter set to 2. We then fit our data to the KMeans object using the fit method.

# Finally, we print the labels assigned to each data point by the KMeans algorithm using the labels_ attribute and the centroids of the two clusters using the cluster_centers_ attribute.

# Note: The scikit-learn implementation of the KMeans algorithm uses the k-means++ initialization method by default, which selects the initial cluster centers in a way that speeds up convergence.