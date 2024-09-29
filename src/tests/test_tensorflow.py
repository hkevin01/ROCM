import tensorflow as tf  # Import TensorFlow library for building and training models
from keras.api.datasets import mnist  # Import the MNIST dataset from the Keras API
from keras.api.layers import Dense, Flatten, Dropout  # Import necessary layer types
from keras.api.models import Sequential  # Import Sequential model type for building a neural network
from keras.api.losses import SparseCategoricalCrossentropy  # Import the loss function for multi-class classification

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# This loads the MNIST dataset, which consists of handwritten digits (0-9).
# `x_train` and `y_train` are the training images and labels,
# while `x_test` and `y_test` are the testing images and labels.

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0
# Pixel values range from 0 to 255. Dividing by 255.0 normalizes the data to a range of 0 to 1,
# which helps improve training convergence.

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 1D arrays of size 784
    Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    Dropout(0.2),  # Dropout layer to reduce overfitting (20% of the neurons will be dropped during training)
    Dense(10)  # Output layer with 10 units (one for each class: digits 0-9)
])

# Make predictions
predictions = model(x_train[:1]).numpy()  # Get predictions for the first training example
print(tf.nn.softmax(predictions).numpy())  # Apply softmax to convert logits to probabilities and print them

# Define the loss function
loss_fn = SparseCategoricalCrossentropy(from_logits=True)  # Use the Sparse Categorical Crossentropy loss function
# `from_logits=True` indicates that the output is not passed through a softmax layer.

# Print loss for the first example
print(loss_fn(y_train[:1], predictions).numpy())  # Calculate and print the loss for the first training example

# Compile the model
model.compile(optimizer='adam',  # Use the Adam optimizer
              loss=loss_fn,  # Set the loss function
              metrics=['accuracy'])  # Track accuracy as a metric during training

# Train the model
model.fit(x_train, y_train, epochs=5)  # Train the model for 5 epochs using the training data
# This is a critical step in the training process. Here's what happens during each epoch:

# Explanation of Epochs in Training
"""
An epoch is one complete pass through the entire training dataset. 
During each epoch, the following occurs:

1. **Forward Pass**: The model makes predictions on the training data.
   - For each batch of data, it calculates the predicted outputs (logits).

2. **Loss Calculation**: The loss function computes how well the model's predictions match the true labels.
   - This quantifies the model's performance.

3. **Backward Pass (Backpropagation)**: 
   - The gradients of the loss with respect to the model's weights are calculated.
   - This helps determine how the weights need to be adjusted to minimize the loss.

4. **Weight Update**: The optimizer (Adam in this case) updates the model's weights based on the gradients.
   - This step aims to reduce the loss for the next iteration.

5. **Metrics Evaluation**: After each epoch, the model's performance metrics (like accuracy) are evaluated.
   - This allows you to track improvement over epochs.

The `epochs` parameter specifies how many times the learning algorithm will work through the entire training dataset.
Thus, setting `epochs=5` means the model will go through the training data five times.
"""

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)  # Evaluate the model on the test dataset and print the results