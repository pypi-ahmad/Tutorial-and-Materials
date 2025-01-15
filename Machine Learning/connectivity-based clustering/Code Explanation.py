from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=50, centers=3, random_state=42)

# Create connectivity matrix based on distance between points
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# Perform clustering using connectivity matrix
clustering = AgglomerativeClustering(n_clusters=3, connectivity=connectivity).fit(X)

# Visualize clustering results
plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
plt.show()


# In this example, we first generate sample data using the make_blobs function from scikit-learn. We then create a connectivity matrix based on the distance between points using the kneighbors_graph function. Finally, we perform connectivity-based clustering using the AgglomerativeClustering function, specifying the number of clusters and the connectivity matrix as inputs. The resulting clusters are visualized using a scatter plot.

# Note that the distance metric used and the number of neighbors in the connectivity matrix can be adjusted to suit the specific needs of the data. Additionally, other algorithms such as DBSCAN and spectral clustering can also be used for connectivity-based clustering.