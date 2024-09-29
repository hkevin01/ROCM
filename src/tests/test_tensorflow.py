import tensorflow as tf
from keras.api.datasets import mnist
from keras.api.layers import Dense, Flatten, Dropout
from keras.api.models import Sequential
from keras.api.losses import SparseCategoricalCrossentropy

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10)
])

# Make predictions
predictions = model(x_train[:1]).numpy()
print(tf.nn.softmax(predictions).numpy())

# Define the loss function
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())

# Compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)