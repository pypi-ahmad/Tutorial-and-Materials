from keras.models import Sequential
from keras.layers import Dense

# Define the MLP model
model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model with appropriate loss function, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# In this example, we first import the necessary modules from Keras. We then define the MLP model using the Sequential API, which allows us to stack layers of the neural network on top of each other. In this case, we add two Dense layers with 32 and 10 neurons respectively. The input_dim parameter specifies the input shape of the data, which in this case is a flattened 28x28 pixel image (784 total input neurons). We also specify the activation functions to use for each layer (relu and softmax).

# Next, we compile the model using the compile function, which takes the desired loss function, optimizer, and metrics as arguments. In this case, we use categorical_crossentropy as the loss function (appropriate for multiclass classification tasks), adam as the optimizer (a popular choice for many neural networks), and accuracy as the metric to monitor during training.

# Finally, we train the model using the fit function, which takes the training data, desired number of epochs, batch size, and validation data as arguments. During training, the model will adjust the weights of the neurons in order to minimize the loss function and improve accuracy on the validation data.