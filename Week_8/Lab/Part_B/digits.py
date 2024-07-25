import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the image containing digits
img = cv2.imread('../Data/digits.png', cv2.IMREAD_GRAYSCALE)

# Split the image into 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# Convert the list of cells into a Numpy array of shape (50, 100, 20, 20)
x = np.array(cells)

# Prepare the labels (0-9 repeated for each row)
labels = np.repeat(np.arange(10), 500)

# Preprocess the data: reshape and normalize
x = x.reshape(-1, 20, 20, 1).astype('float32') / 255.0

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))

# Save the trained model
model.save('../Data/digits_recognition_model.h5')

# Plotting code
plt.style.use('seaborn-darkgrid')

# Create a figure with specified size
plt.figure(figsize=(16, 8))

# Extract epochs range
epochs_range = range(1, 26)

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.legend(loc='lower right', fontsize='large')
plt.title('Training and Validation Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss', marker='o')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss', marker='o')
plt.legend(loc='upper right', fontsize='large')
plt.title('Training and Validation Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Display the plots
plt.tight_layout()
plt.show()