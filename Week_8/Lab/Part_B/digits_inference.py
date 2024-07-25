import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to preprocess the image and find contours
def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours, thresh1

# Main function to capture video and predict digits
def main():
    # Load the pre-trained model
    loaded_model = load_model('../Data/digits_recognition_model.h5')

    # Start capturing video
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        img, contours, thresh = get_img_contour_thresh(img)
        ans1 = ''

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (20, 20))  # Resize to 20x20 pixels
                newImage = np.array(newImage)
                newImage = newImage.reshape(1, 20, 20, 1)
                newImage = newImage.astype('float32') / 255
                prediction = loaded_model.predict(newImage)
                ans1 = np.argmax(prediction)

        x, y, w, h = 0, 0, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Prediction : " + str(ans1), (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
