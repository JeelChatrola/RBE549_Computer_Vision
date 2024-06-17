import cv2
import datetime
import numpy as np
import sys
import os
import feature_match as fm
from pano_wrapper import PanoramaStitching


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
            'Sobel-X': False,
            'Sobel-Y': False,
            'Canny': False,
            'Object-Detection': False,
            'Panorama': False
        }

        # Create a window
        cv2.namedWindow('Camera')

        # Create trackbar for zoom level
        cv2.createTrackbar('Zoom', 'Camera', 0, 100, lambda x: None)
        cv2.createTrackbar('Blur-SigmaX', 'Camera', 5, 30, lambda x: None)
        cv2.createTrackbar('Blur-SigmaY', 'Camera', 5, 30, lambda x: None)
        cv2.createTrackbar('Sobel-X', 'Camera', 0, 15, lambda x: None)
        cv2.createTrackbar('Sobel-Y', 'Camera', 0, 15, lambda x: None)
        cv2.createTrackbar('Canny-1', 'Camera', 1, 5000, lambda x: None)
        cv2.createTrackbar('Canny-2', 'Camera', 1, 5000, lambda x: None)

        self.key = -1
        self.pano_image = 0

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
        sys.stdout.write("  'o' to perform object detection\n")
        sys.stdout.write("  'p' to perform panorama stitching\n")
        sys.stdout.write("  's+x' to SobelX the image\n")
        sys.stdout.write("  's+y' to SobelY the image\n")
        sys.stdout.write("  'd' to Canny Edge the image\n")
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
    
    def convolve(self,img, kernel):
        return cv2.filter2D(img, -1, kernel)
    
    def custom_sobel_x(self,img):
        kernel = np.array([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]])
        return self.convolve(img, kernel)

    def custom_sobel_y(self,img):
        kernel = np.array([[-1, -2, -1], 
                           [0, 0, 0], 
                           [1, 2, 1]])
        return self.convolve(img, kernel)

    def custom_laplacian(self,img):
        kernel = np.array([[0, 1, 0], 
                           [1, -4, 1], 
                           [0, 1, 0]])
        return self.convolve(img, kernel)

    def process_frame(self, frame):
        ######### Zoom ####################################### 
        # Get the zoom level from the trackbar
        # zoom_level = 1 + cv2.getTrackbarPos('Zoom', 'Camera')/10
        # Zoom the frame
        # frame = self.zoom_center(frame,zoom_level)

        ######### Processing ################################        
        # Convert to elif for stopping combination which creates more complicated features
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

        if self.features['Sobel-X']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel_size = cv2.getTrackbarPos('Sobel-X', 'Camera')
            frame = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=2*kernel_size + 1)

        if self.features['Sobel-Y']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel_size = cv2.getTrackbarPos('Sobel-Y', 'Camera')
            frame = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=2*kernel_size + 1)
        
        if self.features['Canny']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Canny(frame, cv2.getTrackbarPos('Canny-1', 'Camera'), cv2.getTrackbarPos('Canny-2', 'Camera'))

        if self.features['Object-Detection']:
            # Load the image
            image = cv2.imread('image_2.png')            
            # Call the sift_flann_match function from feature_match.py
            frame = fm.sift_flann_match(frame, image)

        if self.features['Panorama']:
            key = cv2.waitKey(1)

            # Capture Images one by one and save them
            if key == ord('a'):
                print('Capturing Image')
                cv2.imwrite('{0}.jpg'.format(self.pano_image,1), frame)
                self.pano_image += 1

            elif key == ord('q'):
                self.pano_image = 0
            
                # Set path to the saved images and create Matches and Pano folders
                InputPath = os.getcwd()
                os.makedirs(InputPath+'/Matches', exist_ok=True)
                os.makedirs(InputPath+'/Pano', exist_ok=True)

                # Create an object of the class pano_wrapper
                pano = PanoramaStitching()

                # Set variables for the pano class
                pano.InputPath = InputPath
                pano.scale = 1
                pano.vis = False

                # Call the function to stitch the images
                pano.image_original = pano.readImages()
                H = pano.match_features_homography()
                pano.Pano(H)
                pano.recursive()

                panoimg = cv2.imread(InputPath + '/Pano/Final_Panorama.jpg')
                cv2.imshow('Pano-Image', panoimg)
                cv2.waitKey(0)
                # Destroy the window
                cv2.destroyWindow('Pano-Image')

                self.features['Panorama'] = False

        return frame

    def run(self):
        # Main loop to keep the video stream running
        while self.cap.isOpened():
            _, frame = self.cap.read()
            frame = self.process_frame(frame)
            cv2.imshow('Camera', frame)

            last_key = self.key
            self.key = cv2.waitKey(1)

            # Print Options and Features
            self.print_features()

            # Capture image 
            if self.key == ord('c'):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                cv2.imwrite(f'Images/IMG_{timestamp}.png', frame)

                white_image = np.full((frame.shape[0], frame.shape[1], 3), 255, dtype=np.uint8)
                cv2.imshow('Camera', white_image)
                cv2.waitKey(200)

            # Record video
            elif self.key == ord('v'):
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
            elif self.key == ord('e'):
                self.features['extract_color'] = not self.features['extract_color']
                
            # Rotate Image
            elif self.key == ord('r'):
                self.features['rotate'] = not self.features['rotate']
                
            # Threshold Image
            elif self.key == ord('t'):
                self.features['threshold'] = not self.features['threshold']
                
            # Blur Image
            elif self.key == ord('b'):
                self.features['blur'] = not self.features['blur']   
            
            # Sobel-X Image
            elif self.key == ord('x') and last_key == ord('s'):
                self.features['Sobel-X'] = not self.features['Sobel-X'] 
            
            # Sobel-Y Image
            elif self.key == ord('y') and last_key == ord('s'):
                self.features['Sobel-Y'] = not self.features['Sobel-Y']

            # Canny Edge Image
            elif self.key == ord('d'):
                self.features['Canny'] = not self.features['Canny']

            # Object Detection
            elif self.key == ord('o'):
                self.features['Object-Detection'] = not self.features['Object-Detection']

            # Panorama
            elif self.key == ord('p'):
                self.features['Panorama'] = not self.features['Panorama'] 

                # Delete the clear images from previous panorama
                path = os.getcwd()+'/Matches/'
                for f in os.listdir(path):
                    if f.endswith('.jpg'):
                        os.remove(path + f)

                path = os.getcwd()+'/Pano/'
                for f in os.listdir(path):
                    if f.endswith('.jpg'):
                        os.remove(path + f)
                
                path = os.getcwd() +'/'
                for f in os.listdir(path):
                    if f[0].isdigit() and f.endswith('.jpg'):
                        os.remove(path + f)               

            # Quit the program
            elif self.key == 27:  # ESC key
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