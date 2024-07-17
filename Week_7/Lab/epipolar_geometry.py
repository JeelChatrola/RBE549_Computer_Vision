import numpy as np
import cv2
from matplotlib import pyplot as plt

image_combo = 2

if image_combo == 1:
    img1 = cv2.imread('images/part3/globe_left.jpg', cv2.IMREAD_GRAYSCALE) #queryimage # left image
    img2 = cv2.imread('images/part3/globe_center.jpg', cv2.IMREAD_GRAYSCALE) #trainimage # right image
elif image_combo == 2:
    img1 = cv2.imread('images/part3/globe_center.jpg', cv2.IMREAD_GRAYSCALE) #queryimage # left image
    img2 = cv2.imread('images/part3/globe_right.jpg', cv2.IMREAD_GRAYSCALE) #trainimage # right image

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''

    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)

    return img1,img2

def compare_features(img1,img2):
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 2
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2
    

pts1, pts2 = compare_features(img1,img2)

# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS) 
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT) 
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Find epilines corresponding to points in images
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
 
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
 
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
if image_combo == 1:
    plt.savefig('images/part3/epipolar_lines_left_center.jpg', dpi=300)  # Increase DPI for higher resolution
elif image_combo == 2:
    plt.savefig('images/part3/epipolar_lines_center_right.jpg', dpi=300)  # Increase DPI for higher resolution
plt.show()



