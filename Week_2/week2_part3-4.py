import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('unity_hall.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Original Image', image)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
rows,cols,ch = image.shape

####################################################################################
# Part 3
####################################################################################

# Rotation
image_center = tuple(np.array(image.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(image_center, 10, 1.0)
abs_cos = abs(rot_mat[0,0])
abs_sin = abs(rot_mat[0,1])
bound_w = int(image.shape[0]*abs_sin + image.shape[1]*abs_cos)
bound_h = int(image.shape[0]*abs_cos + image.shape[1]*abs_sin)
rot_mat[0, 2] += bound_w/2 - image_center[0]
rot_mat[1, 2] += bound_h/2 - image_center[1]
rotated_image = cv2.warpAffine(image, rot_mat, (bound_w, bound_h))

# Scaled Up
scaled_up_image = cv2.resize(image,(int(1.2*cols),int(1.2*rows)), interpolation = cv2.INTER_CUBIC)

# Scaled Down
scaled_down_image = cv2.resize(image,(int(0.8*cols),int(0.8*rows)), interpolation = cv2.INTER_CUBIC)

# Affine transformation
src_points = np.float32([[50,50],[200,50],[50,200]])
dst_points = np.float32([[50,50],[230,10],[45,220]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
transformed_corners = cv2.transform(np.float32([[ [0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0] ]]), affine_matrix)
min_x, min_y = np.intp(transformed_corners.reshape(-1, 2).min(axis=0))
max_x, max_y = np.intp(transformed_corners.reshape(-1, 2).max(axis=0))
affine_matrix[0, 2] -= min_x
affine_matrix[1, 2] -= min_y
affine_image = cv2.warpAffine(image, affine_matrix, (max_x - min_x, max_y - min_y))

# Perspective transformation
src_points = np.float32([[100,100],[250,100],[100,250],[250,250]])
dst_points = np.float32([[90,75],[270,90],[90,230],[260,260]])
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
transformed_corners = cv2.perspectiveTransform(np.float32([[ [0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0] ]]), perspective_matrix)
min_x, min_y = np.intp(transformed_corners.reshape(-1, 2).min(axis=0))
max_x, max_y = np.intp(transformed_corners.reshape(-1, 2).max(axis=0))
translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
adjusted_matrix = np.dot(translation_matrix, perspective_matrix)
perspective_image = cv2.warpPerspective(image, adjusted_matrix, (max_x - min_x, max_y - min_y))

# Display images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.imshow('Scaled Up Image', scaled_up_image)
cv2.imshow('Scaled Down Image', scaled_down_image)
cv2.imshow('Affine Image', affine_image)
cv2.imshow('Perspective Image', perspective_image)

####################################################################################
# Part 4
####################################################################################

# Harris Corner Detection
def harris_corner_detection(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.09)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst>0.01*dst.max()]=[0,0,255]
    return image

# Apply harris to the above 6 images
harris_image = harris_corner_detection(image)
harris_rotated_image = harris_corner_detection(rotated_image)
harris_scaled_up_image = harris_corner_detection(scaled_up_image)
harris_scaled_down_image = harris_corner_detection(scaled_down_image)
harris_affine_image = harris_corner_detection(affine_image)
harris_perspective_image = harris_corner_detection(perspective_image)

# Display images
plt.figure(figsize=(10, 6))

plt.subplot(231), plt.imshow(cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB)), plt.title('Harris - Original Image'), plt.axis('off')
plt.subplot(232), plt.imshow(cv2.cvtColor(harris_rotated_image, cv2.COLOR_BGR2RGB)), plt.title('Harris - Rotated Image'), plt.axis('off')
plt.subplot(233), plt.imshow(cv2.cvtColor(harris_scaled_up_image, cv2.COLOR_BGR2RGB)), plt.title('Harris - Scaled Up Image'), plt.axis('off')
plt.subplot(234), plt.imshow(cv2.cvtColor(harris_scaled_down_image, cv2.COLOR_BGR2RGB)), plt.title('Harris - Scaled Down Image'), plt.axis('off')
plt.subplot(235), plt.imshow(cv2.cvtColor(harris_affine_image, cv2.COLOR_BGR2RGB)), plt.title('Harris - Affine Image'), plt.axis('off')
plt.subplot(236), plt.imshow(cv2.cvtColor(harris_perspective_image, cv2.COLOR_BGR2RGB)), plt.title('Harris - Perspective Image'), plt.axis('off')

plt.tight_layout()

# SIFT (Scale-Invariant Feature Transform)
def sift_features(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,image,color=(0,255,0))
    return img

# Apply sift to the above 6 images
sift_image = sift_features(image)
sift_rotated_image = sift_features(rotated_image)
sift_scaled_up_image = sift_features(scaled_up_image)
sift_scaled_down_image = sift_features(scaled_down_image)
sift_affine_image = sift_features(affine_image)
sift_perspective_image = sift_features(perspective_image)

# Display images
plt.figure(figsize=(10, 6))

plt.subplot(231), plt.imshow(cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB)), plt.title('SIFT - Original Image'), plt.axis('off')
plt.subplot(232), plt.imshow(cv2.cvtColor(sift_rotated_image, cv2.COLOR_BGR2RGB)), plt.title('SIFT - Rotated Image'), plt.axis('off')
plt.subplot(233), plt.imshow(cv2.cvtColor(sift_scaled_up_image, cv2.COLOR_BGR2RGB)), plt.title('SIFT - Scaled Up Image'), plt.axis('off')
plt.subplot(234), plt.imshow(cv2.cvtColor(sift_scaled_down_image, cv2.COLOR_BGR2RGB)), plt.title('SIFT - Scaled Down Image'), plt.axis('off')
plt.subplot(235), plt.imshow(cv2.cvtColor(sift_affine_image, cv2.COLOR_BGR2RGB)), plt.title('SIFT - Affine Image'), plt.axis('off')
plt.subplot(236), plt.imshow(cv2.cvtColor(sift_perspective_image, cv2.COLOR_BGR2RGB)), plt.title('SIFT - Perspective Image'), plt.axis('off')

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()