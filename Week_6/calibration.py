#!/usr/bin/env python3
 
import cv2
import numpy as np
import os
import glob

vis = False
np.set_printoptions(precision=3)

# Part 1
folder = 'opencv_calibration_imgs' 
CHECKERBOARD = (6,9)

# Part 2
# folder = 'prof_calibration_imgs'
# CHECKERBOARD = (7,11)

# Part 3
# folder = 'custom_calibration_imgs' 
# CHECKERBOARD = (6,9)

# Defining the dimensions of checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
 
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob(os.path.join(folder, '*.jpg'))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    if vis:
        cv2.imshow('img',img)
        cv2.waitKey(0)
 
cv2.destroyAllWindows()
 
h,w = img.shape[:2]
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera calibration result for later use
print("Camera matrix : \n")
print(mtx)
np.save(folder + '/camera_matrix.npy', mtx)
print("")

print("dist : \n")
print(dist)
np.save(folder + '/camera_dist_coeff.npy', dist)
print("")

print("rvecs : \n")
print(rvecs)
np.save(folder + '/camera_rvecs.npy', rvecs)
print("")

print("tvecs : \n")
print(tvecs)
np.save(folder + '/camera_tvecs.npy', tvecs)
print("")

# Re-projection error gives a good estimation of just how exact the found parameters are.
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

# Reprojection error save
np.save(folder + '/camera_reprojection_error.npy', mean_error)


# Get new image which are undistorted using the camera matrix and distortion coefficients
for fname in images:
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
 
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(folder + '/calibresult/' + fname.split('/')[-1], dst)

    if vis:
        cv2.imshow('img',dst)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()