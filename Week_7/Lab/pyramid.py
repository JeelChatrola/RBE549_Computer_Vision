import numpy as np
import cv2 as cv
import os
import glob

# Load previously saved data
script_dir = os.path.dirname(os.path.abspath(__file__))
camera_matrix_path = os.path.join(script_dir, 'images', 'part1', 'camera_matrix.npy')
camera_dist_coeff_path = os.path.join(script_dir, 'images', 'part1', 'camera_dist_coeff.npy')

# Load MTX
mtx = np.load(camera_matrix_path)

# Load DIST
dist = np.load(camera_dist_coeff_path)

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    
    # draw pillars in blue color
    j=4
    for i in zip(range(4)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255,0,0),3)
    
    # draw top 
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

# Defining the criteria for ending
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Axis for the pyramid
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[1.5,1.5,-3] ])
i=0

for fname in glob.glob('images/part1/left*.jpg'):
    
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)
    
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
    
        # print(corners2.shape,imgpts.shape)
        img = draw(img,corners2,imgpts)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        
        # Save the image
        if k == ord('s'):
            cv.imwrite('images/part1/results/triangular'+str(i)+'.jpg', img)

    i+=1

cv.destroyAllWindows()