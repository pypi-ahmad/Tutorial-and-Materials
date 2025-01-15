from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Define the generator model
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(28,28,1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the combined generator and discriminator model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Load the MNIST dataset
from keras.datasets import mnist
(X_train, _), (_, _) = mnist.load_data()

# Normalize the pixel values to the range [-1, 1]
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

# Set the number of epochs and the size of the latent space
epochs = 100
latent_dim = 100

# Build and compile the models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Train the GAN
batch_size = 128
half_batch = int(batch_size / 2)
for epoch in range(epochs):
    # Train the discriminator
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_images = X_train[idx]
    noise = np.random.randn(half_batch, latent_dim)
    fake_images = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train the generator
    noise = np.random.randn(batch_size, latent_dim)
    y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, y)
    
    # Print the progress
    # Print the progress
    print(f"Epoch {epoch+1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

    # Save generated images every 10 epochs
    if (epoch+1) % 10 == 0:
        noise = np.random.randn(25, latent_dim)
        gen_images = generator.predict(noise)
        gen_images = 0.5 * gen_images + 0.5  # Scale images back to the range [0, 1]
        
        # Plot the generated images
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(gen_images[count, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                count += 1
        plt.show()
