import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
#Loading and normalizing the data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Defining the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#Defining the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizers = {
    'SGD': SGD(),
    'Adam': Adam(),
    'RMSprop': RMSprop()
}

results = {}

for optimizer_name, optimizer in optimizers.items():

  model.compile(optimizer=optimizer_name,
              loss=loss_fn,
              metrics=['accuracy'])

  print(f'Training with optimizer: {optimizer_name}')
  history = model.fit(x_train, y_train, epochs=5)
  results[optimizer_name] = history

# Set the style for plots
plt.style.use('seaborn-darkgrid')

# Plot training accuracy and loss graphs
plt.figure(figsize=(12, 6))

for optimizer_name, history in results.items():
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label=f'{optimizer_name} Accuracy', marker='o')  # Added marker
    plt.title('Training Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label=f'{optimizer_name} Loss', marker='o')  # Added marker
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()

plt.tight_layout()
plt.savefig("../Data/optimizer_comparison_custom.png")
plt.show()

# Report results
for optimizer_name, history in results.items():
    print(f'\nOptimizer: {optimizer_name}')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')