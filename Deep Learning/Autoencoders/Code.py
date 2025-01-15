from keras.layers import Input, Dense
from keras.models import Model

# Define input shape
input_shape = (784,)

# Define encoder architecture
inputs = Input(shape=input_shape)
encoded = Dense(128, activation='relu')(inputs)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# Define decoder architecture
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# Define autoencoder model
autoencoder = Model(inputs, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True)


# In this example, we are training an autoencoder to reconstruct images from the MNIST dataset. The input shape is (784,) since each image in the dataset is a 28x28 grayscale image flattened into a 1D array of 784 elements. We define the encoder architecture with three dense layers with ReLU activation, reducing the dimensionality of the input data to a 32-dimensional latent space. The decoder architecture mirrors the encoder architecture, but with sigmoid activation in the final layer to output values between 0 and 1. The autoencoder model is compiled with the Adam optimizer and binary cross-entropy loss function, and then trained on the MNIST training set for 10 epochs with a batch size of 256.