import cv2
import tensorflow as tf
import numpy as np

# Class names corresponding to your model's output labels
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Load the pre-trained model
model = tf.keras.models.load_model('../Data/flower_trained_h5.h5', compile=False)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Define the region of interest (ROI) where the image will be captured and classified
    x, y, w, h = 50, 50, 180, 180
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the box

    # Extract the region of interest for classification
    roi = frame[y:y+h, x:x+w]
    img = cv2.resize(roi, (180, 180))  # Resize the ROI to match the model's input size

    # Prepare the image for the model
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Predict the class of the flower
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display the classification result
    result_text = "This image most likely belongs to {} with a {:.2f}% confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score))
    print(result_text)

    # Display the result on the video frame
    cv2.putText(frame, class_names[np.argmax(score)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Flower Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
