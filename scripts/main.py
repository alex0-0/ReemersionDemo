#!/usr/bin/env python3
""" Call functions to work, program entry """
import cv2
import detect
import numpy as np
import distort
import match

img = cv2.imread("j.png")
img_1 = cv2.imread("affine_2.png")

#extract feature points
kp1, des1 = detect.extractFeatures(img)
kp2, des2 = detect.extractFeatures(img_1)

#match feature points
matches = match.matchFeature(des1, kp1, des2, kp2)
# Draw matches.
#img3 = cv2.drawMatchesKnn(img,kp1,img_1,kp2,matches, None, flags=2)
img3 = cv2.drawMatches(img,kp1,img_1,kp2,matches, None, flags=2)

cv2.imshow("match", img3)
cv2.waitKey(0)

t = detect.extractDistinctFeatures(img)
for i in t:
    cv2.imshow("distorted images",i)
    cv2.waitKey(0)

cv2.destroyAllWindows()

kps, des = detect.extractFeatures(img)

img_kp = cv2.drawKeypoints(img, kps, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("image with key points", img_kp)
cv2.waitKey(0)

#save distorted image to default directory, i.e., distorted_img
distort.saveDistortedImages(img)
