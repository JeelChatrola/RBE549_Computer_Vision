import tensorflow as tf
from tensorflow.keras.applications import VGG19, InceptionV3 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os


def load_and_prepare_data(data_dir, img_size=(227, 227), batch_size=32, validation_split=0.2):
    # Load all images and labels
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_indices[class_name])

    images = np.array(images)
    labels = np.array(labels)

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=len(class_names))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=validation_split, stratify=labels)

    # Create ImageDataGenerators with more augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    validation_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")

    return train_generator, validation_generator

def create_model(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)  # Apply dropout to the output of the dense layer
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def train_model(model, train_generator, validation_generator, epochs=10):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)

    print(f"Training steps per epoch: {steps_per_epoch}")
    print(f"Validation steps per epoch: {validation_steps}")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
    )

    return history

def test_random_images(model, data_dir, img_size=(227, 227), num_images=2):
    class_names = sorted(os.listdir(data_dir))
    for _ in range(num_images):
        random_class = np.random.choice(class_names)
        random_image_path = np.random.choice(os.listdir(os.path.join(data_dir, random_class)))
        img_path = os.path.join(data_dir, random_class, random_image_path)
        
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        print(f"True class: {random_class}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
        print()

def plot_results(history, model_name):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{model_name}_training_plot.png'
    plt.savefig("../RBE549_Computer_Vision/Week_9/Lab/Data/results/"+plot_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Plot saved as {plot_filename}")

def main():
    # Specify the correct path to the MerchData directory
    data_direc = os.path.dirname(os.path.abspath(__file__))
    data_dir1 = '../Data/MerchData'

    data_dir = os.path.join(data_direc, data_dir1)

    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Resolve the absolute path of the data directory
    abs_data_dir = os.path.abspath(data_dir)
    print(f"Resolved data directory path: {abs_data_dir}")

    # Check if the directory exists
    if not os.path.isdir(abs_data_dir):
        print(f"Error: The directory {abs_data_dir} does not exist.")
        print("Please ensure the MerchData directory is in the correct location.")
        return

    print(f"Using data directory: {abs_data_dir}")

    img_size = (224, 224)
    batch_size = 32
    epochs = 10

    try:
        num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if num_classes == 0:
            raise ValueError("No subdirectories found in the data directory.")
    except Exception as e:
        print(f"Error while reading the data directory: {e}")
        return

    print(f"Number of classes detected: {num_classes}")

    train_generator, validation_generator = load_and_prepare_data(data_dir, img_size, batch_size)

    # VGG19
    base_model_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    model_vgg19 = create_model(base_model_vgg19, num_classes)
    print("Training VGG19 model...")
    history_vgg19 = train_model(model_vgg19, train_generator, validation_generator, epochs)
    print("VGG19 training history keys:", history_vgg19.history.keys())

    print("\nVGG19 Results:")
    plot_results(history_vgg19, 'VGG19')
    test_random_images(model_vgg19, data_dir, img_size)

    # Reset generators
    train_generator.reset()
    validation_generator.reset()

    # InceptionV3
    base_model_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    model_inception = create_model(base_model_inception, num_classes)
    print("\nTraining InceptionV3 model...")
    history_inception = train_model(model_inception, train_generator, validation_generator, epochs)
    print("InceptionV3 training history keys:", history_inception.history.keys())

    print("\nInceptionV3 Results:")
    plot_results(history_inception, 'InceptionV3')
    test_random_images(model_inception, data_dir, img_size)

    # Compare performance
    print("\nPerformance Comparison:")
    print("VGG19 final validation accuracy:", history_vgg19.history['val_accuracy'][-1])
    print("InceptionV3 final validation accuracy:", history_inception.history['val_accuracy'][-1])

    print("\nTesting VGG19 on random images:")
    test_random_images(model_vgg19, data_dir, img_size)

    print("\nTesting InceptionV3 on random images:")
    test_random_images(model_inception, data_dir, img_size)

if __name__ == "__main__":
    main()