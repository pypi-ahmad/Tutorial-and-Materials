import numpy as np
from minisom import MiniSom

# load data
data = np.loadtxt('data.txt')

# create SOM object
som = MiniSom(x=10, y=10, input_len=4, sigma=1.0, learning_rate=0.5)

# initialize weights
som.random_weights_init(data)

# train SOM
som.train_random(data, 100)

# visualize SOM results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

# mark anomalies in red
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(data):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor='None', markersize=10, markeredgewidth=2)
show()


# In this code, we first load our data using the np.loadtxt function. We then create a SOM object using the MiniSom class, specifying the number of nodes in the x and y dimensions (x=10, y=10), the number of input features (input_len=4), the neighborhood radius (sigma=1.0), and the initial learning rate (learning_rate=0.5).

# Next, we initialize the weights of the SOM using the random_weights_init method, and then train the SOM using the train_random method for a specified number of iterations (100 in this case).

# Finally, we visualize the SOM using the distance_map method to create a color-coded map of the nodes based on their similarity, and then mark any anomalies in the data with red markers using the winner method to identify the best-matching node for each data point.