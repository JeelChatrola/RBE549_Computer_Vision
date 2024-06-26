import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_detector(img): 
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def surf_detector(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    return kp, des

def FLANN_matcher(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def BF_matcher(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    return matches

def sift_bf_match(img1, img2):
    kp1, des1 = sift_detector(img1)
    kp2, des2 = sift_detector(img2)
    
    matches = BF_matcher(des1, des2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    return img3, good
    
def sift_flann_match(img1, img2):
    kp1, des1 = sift_detector(img1)
    kp2, des2 = sift_detector(img2)

    matches = FLANN_matcher(des1, des2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
        
    matchesMask = [m for m, n in matches if m.distance < 0.75 * n.distance]
     
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None)

    return img3, matchesMask

def surf_bf_match(img1, img2):
    kp1, des1 = surf_detector(img1)
    kp2, des2 = surf_detector(img2)

    matches = BF_matcher(des1, des2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    
    return img3, good

def surf_flann_match(img1, img2):
    kp1, des1 = surf_detector(img1)
    kp2, des2 = surf_detector(img2)

    matches = FLANN_matcher(des1, des2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
     
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,1]
    
    draw_params = dict(matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        matchesMask = matchesMask,
        flags = cv2.DrawMatchesFlags_DEFAULT)
     
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    return img3, matchesMask

