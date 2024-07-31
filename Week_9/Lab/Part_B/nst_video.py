import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

# Load the style transfer model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the style image
style_image_name = '../Data/Vassily.jpg'  # Replace with your style image filename
style_image_path = os.path.join(script_dir, style_image_name)

print(f"Attempting to load style image from: {style_image_path}")

if not os.path.exists(style_image_path):
    print(f"Error: The style image file '{style_image_name}' does not exist in the script directory.")
    print(f"Current script directory: {script_dir}")
    print("Please make sure you've placed the style image in the same folder as this script.")
    print("Also, check that the filename is correct, including the extension.")
    exit(1)

style_image = cv2.imread(style_image_path)
if style_image is None:
    print(f"Error: Unable to read the style image file '{style_image_name}'.")
    print("Please make sure the file is a valid image format (e.g., JPEG, PNG).")
    exit(1)

style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.0
style_image = tf.convert_to_tensor(style_image)  # Convert to TensorFlow tensor

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32)[np.newaxis, ...] / 255.0
    return tf.convert_to_tensor(frame)  # Convert to TensorFlow tensor

# Function to postprocess the stylized frame
def postprocess_frame(frame):
    frame = frame.numpy()  # Convert TensorFlow tensor to NumPy array
    frame = np.squeeze(frame)
    frame = np.clip(frame, 0, 1) * 255
    return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the webcam.")
    print("Please make sure your webcam is connected and not being used by another application.")
    exit(1)

print("Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Apply style transfer
    stylized_frame = model(input_frame, style_image)[0]

    # Postprocess the stylized frame
    output_frame = postprocess_frame(stylized_frame)

    # Display the original and stylized frames
    cv2.imshow('Original', frame)
    cv2.imshow('Stylized', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting the application.")
        break

cap.release()
cv2.destroyAllWindows()