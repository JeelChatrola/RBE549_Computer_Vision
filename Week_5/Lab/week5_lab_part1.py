import cv2
import numpy as np
import feature_match as fm

def read_coin_images():
    """
    Reads coin images from the specified directory and returns them as a list.
    
    Returns:
        list: A list of coin images in BGR format.
    """
    penny = cv2.imread("Images/coins_my/template/penny.jpg", cv2.IMREAD_COLOR)  
    quarter = cv2.imread("Images/coins_my/template/quarter.jpg", cv2.IMREAD_COLOR) 
    dime = cv2.imread("Images/coins_my/template/dime.jpg", cv2.IMREAD_COLOR)
    nickel = cv2.imread("Images/coins_my/template/nickel.jpg", cv2.IMREAD_COLOR)
    coin_images = [penny, quarter, dime, nickel]
    return coin_images

def detect_coin_value(frame):
    """
    Detects coins in the given frame and estimates their value based on template matching.
    
    Args:
        frame (numpy.ndarray): The frame in which coins are to be detected.
    
    Returns:
        tuple: A tuple containing the total dollar value of detected coins and the frame with detected coins highlighted.
    """

    detected_circles = hough_circle_detection(frame)
    coin_images = read_coin_images()
    frame_circle = frame.copy()
    
    penny_count = 0
    quarter_count = 0
    dime_count = 0  
    nickel_count = 0

    if detected_circles is not None:        
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
            roi = frame[b-r:b+r, a-r:a+r]
            cv2.circle(frame_circle, (a, b), r, (0, 255, 255), 2) 
            
            _, good_quarter = fm.sift_flann_match(coin_images[1],roi)
            _, good_penny = fm.sift_flann_match(coin_images[0],roi)
            _, good_dime = fm.sift_flann_match(coin_images[2],roi)
            _, good_nickel = fm.sift_flann_match(coin_images[3],roi)

            matches = np.array([len(good_penny), len(good_quarter), len(good_dime), len(good_nickel)])
            idx = np.argmax(matches)
            if sum(matches)==0:
                continue

            if idx == 0:
                penny_count += 1
            elif idx == 1:
                quarter_count += 1
            elif idx == 2:
                dime_count += 1
            elif idx == 3:
                nickel_count += 1
    
    dollar_value = (penny_count * 0.01) + (quarter_count * 0.25) + (dime_count * 0.1) + (nickel_count * 0.05)

    return round(dollar_value,2), frame_circle

def hough_circle_detection(frame):
    """
    Detects circles in the given frame using the Hough Circle Transform.
    
    Args:
        frame (numpy.ndarray): The frame in which circles are to be detected.
    
    Returns:
        numpy.ndarray: An array of detected circles.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 3)
    detected_circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, gray.shape[0]/8, param1=50, param2=40, minRadius=30, maxRadius=120)
    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles))    
        
    return detected_circles

def process_frame(frame):
    """
    Processes a single frame to detect coins, calculate their total value, and display the value on the frame.
    
    Args:
        frame (numpy.ndarray): The frame to process.
    
    Returns:
        numpy.ndarray: The processed frame with the total value of detected coins displayed.
    """
    dollar, frame = detect_coin_value(frame)
    frame = cv2.putText(frame, str(dollar)+'$', (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return frame

def main():
    """
    Main function to capture video from a camera, process each frame, and display the result.
    """
    # using /dev/video2 for my webcam (phone camera)
    cap = cv2.VideoCapture('/dev/video2') # 0 for default web camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = process_frame(frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()