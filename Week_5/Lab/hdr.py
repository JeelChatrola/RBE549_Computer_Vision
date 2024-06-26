import cv2 as cv
import numpy as np

img_fn = ["img1.jpg", "img2.jpg", 'img3.jpg']
img_list = [cv.imread(fn) for fn in img_fn]

# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)
