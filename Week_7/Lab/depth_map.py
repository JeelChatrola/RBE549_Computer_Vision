import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
imgL = cv.imread('images/part2/aloeL.jpg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('images/part2/aloeR.jpg', cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM.create(numDisparities=128, blockSize=15)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.imsave('images/part2/disparity_map.jpg',disparity)
plt.show()