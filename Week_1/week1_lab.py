import cv2
import datetime
import numpy as np
import sys
import os

class VideoProcessor:
    def __init__(self, video_source):
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        self.out = None

        # Initialize variables for switching between different options
        self.features = {
            'recording': False,
            'extract_color': False,
            'rotate': False,
            'threshold': False,
            'blur': False,
            'sharpen': False
        }

        # Create a window
        cv2.namedWindow('Camera')

        # Create trackbar for zoom level
        cv2.createTrackbar('Zoom', 'Camera', 0, 100, lambda x: None)
        cv2.createTrackbar('Blur-SigmaX', 'Camera', 5, 30, lambda x: None)
        cv2.createTrackbar('Blur-SigmaY', 'Camera', 5, 30, lambda x: None)

        # Load the OpenCV logo
        self.logo = cv2.imread('OpenCV_Logo.png')
        self.logo = cv2.resize(self.logo, (150, 150))

    def print_features(self):
        os.system('clear')
        sys.stdout.write("Camera Booth: \n")
        sys.stdout.write("\n")
        sys.stdout.write("Press keys to enable/disable features: \n")

        sys.stdout.write("  'c' to Capture an image \n")
        sys.stdout.write("  'v' to start/stop recording a video\n")
        sys.stdout.write("  'e' to extract color\n")
        sys.stdout.write("  'r' to rotate the image\n")
        sys.stdout.write("  't' to threshold the image\n")
        sys.stdout.write("  'b' to blur the image\n")
        sys.stdout.write("  's' to sharpen the image\n")
        sys.stdout.write("  'ESC' to quit the program\n\n")

        sys.stdout.write("Enabled features: ")

        enabled_features = ", ".join(feature for feature, state in self.features.items() if state)
        sys.stdout.write(enabled_features if enabled_features else "None")
        sys.stdout.flush()  # Ensure it's displayed immediately

    def rotate_image(self, image, angle):
        # Get the image center
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        
        # Rotate the image
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        
        # Apply the rotation
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        return result

    def zoom_center(self, img, zoom_factor):
        # Get the image size
        y_size = img.shape[0]
        x_size = img.shape[1]
        
        # Calculate the zoomed image size as per zoom factor
        x1 = int(0.5*x_size*(1-1/zoom_factor))
        x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
        y1 = int(0.5*y_size*(1-1/zoom_factor))
        y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

        # Crop the image
        img_cropped = img[y1:y2,x1:x2]

        return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)

    def process_frame(self, frame):
        ######### Zoom ####################################### 
        # Get the zoom level from the trackbar
        zoom_level = 1 + cv2.getTrackbarPos('Zoom', 'Camera')/10
        # Zoom the frame
        frame = self.zoom_center(frame,zoom_level)

        ######### Red-Border #################################
        # Added a red border to the frame
        frame = cv2.copyMakeBorder(frame,5,5,5,5,cv2.BORDER_CONSTANT,value=[0,0,255])

        ######### Logo-Blend #################################
        # Blend the logo into the top-left corner of the frame
        roi = frame[0:self.logo.shape[0], 0:self.logo.shape[1]]
        blended = cv2.addWeighted(roi, 0.7, self.logo, 0.3, 0)
        frame[0:self.logo.shape[0], 0:self.logo.shape[1]] = blended

        ######### Time-Stamp #################################
        # Add a timestamp to the bottom right corner of the frame
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        # Get the size of the textbox
        text_width, text_height = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        # Put timestamp on the frame bottom right corner
        cv2.putText(frame, timestamp, (frame.shape[1] - text_width - 10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        ######### Top right-ROI ##############################
        margin = 5  # Increase this value to increase the size of the ROI
        # Get the bottom left corner of the timestamp
        bottom_left_corner = (frame.shape[1] - text_width - 10, frame.shape[0] - 10)
        # Select the timestamp as a ROI
        roi = frame[bottom_left_corner[1] - text_height - margin:bottom_left_corner[1] + margin, bottom_left_corner[0] - margin:bottom_left_corner[0] + text_width + margin]
        # Copy the ROI to the top right corner
        frame[10:10 + text_height + 2*margin, frame.shape[1] - text_width - 10 - 2*margin:frame.shape[1] - 10] = roi

        ######### Processing ################################        
        if self.features['extract_color']:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # define range of red color in HSV
            lower_red = np.array([0,140,100])
            upper_red = np.array([180,255,255])

            # Threshold the HSV image to get only red colors
            mask2 = cv2.inRange(hsv, lower_red, upper_red)

            # Bitwise-AND mask and original image
            frame = cv2.bitwise_and(frame,frame, mask=mask2)

        if self.features['rotate']:
            frame = self.rotate_image(frame, 10)

        if self.features['threshold']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)

        if self.features['blur']:
            X = 2 * cv2.getTrackbarPos('Blur-SigmaX', 'Camera') + 1
            Y = 2 * cv2.getTrackbarPos('Blur-SigmaY', 'Camera') + 1
            
            frame = cv2.GaussianBlur(frame, (X, Y), 0)

        if self.features['sharpen']:
            frame = cv2.bilateralFilter(frame,9,75,75)

        return frame

    def run(self):
        # Main loop to keep the video stream running
        while self.cap.isOpened():
            _, frame = self.cap.read()
            frame = self.process_frame(frame)
            cv2.imshow('Camera', frame)

            key = cv2.waitKey(1)

            # Print Options and Features
            self.print_features()

            # Switching between different features
            # Capture image 
            if key == ord('c'):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                cv2.imwrite(f'Images/IMG_{timestamp}.png', frame)

                white_image = np.full((frame.shape[0], frame.shape[1], 3), 255, dtype=np.uint8)
                cv2.imshow('Camera', white_image)
                cv2.waitKey(200)

            # Record video
            elif key == ord('v'):
                if self.features['recording']:
                    # Stop recording
                    self.out.release()
                    self.features['recording'] = False
                else:
                    # Start recording
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
                    frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Use the resolution in the VideoWriter
                    self.out = cv2.VideoWriter(f'Videos/VID_{timestamp}.avi', fourcc, 20.0, (frame_width, frame_height))
                    self.features['recording'] = True

            # Extract Color
            elif key == ord('e'):
                self.features['extract_color'] = not self.features['extract_color']
                
            # Rotate Image
            elif key == ord('r'):
                self.features['rotate'] = not self.features['rotate']
                
            # Threshold Image
            elif key == ord('t'):
                self.features['threshold'] = not self.features['threshold']
                
            # Blur Image
            elif key == ord('b'):
                self.features['blur'] = not self.features['blur']   
                
            # Sharpen Image
            elif key == ord('s'):
                self.features['sharpen'] = not self.features['sharpen']
                
            # Quit the program
            elif key == 27:  # ESC key
                break

            # Write frame to video file if recording
            if self.features['recording']:
                frame_resized = cv2.resize(frame, (640, 480))
                self.out.write(frame_resized)

        # When everything done, release the capture and destroy windows
        self.cap.release()

        if self.out is not None:
            self.out.release()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = VideoProcessor(0)  # 0 is the default/webcam
    processor.run()