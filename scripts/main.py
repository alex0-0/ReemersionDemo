#!/usr/bin/env python3
""" Call functions to work, program entry """
import cv2
import detect
import numpy as np
import distort
import match

DEBUG = True
TAG = "MAIN\t"

def trackFeatureChange(img, angle_step, scale_step, affine_step, pers_step):
    kp, des = detect.extractORBFeatures(img)
    print(TAG + "original: " + str(len(kp)))

    print(TAG + "rotate:")
    t = distort.rotateImage(img)
    count = 0
    for i in t:
        kp, des = detect.extractORBFeatures(i)
        print(TAG + ("+" if (count%2==0) else "-") + str((count//2+1)*angle_step) + ": " + str(len(kp)))
        count += 1

    print(TAG + "scale:")
    t = distort.scaleImage(img)
    count = 0
    for i in t:
        kp, des = detect.extractORBFeatures(i)
        scale = (1 + (count//2+1)*scale_step) if (count%2==0) else (1 - (count//2+1)*scale_step)
        print(TAG + str(scale) + ": " + str(len(kp)))
        count += 1

    print(TAG + "affine:")
    t = distort.affineImage(img)
    count = 1
    for i in t:
        kp, des = detect.extractORBFeatures(i)
        print(TAG + str(count) + ": " + str(len(kp)))
        count += 1

    print(TAG + "perspective:")
    t = distort.changeImagePerspective(img)
    count = 1
    for i in t:
        kp, des = detect.extractORBFeatures(i)
        print(TAG + str(count) + ": " + str(len(kp)))
        count += 1


img = cv2.imread("j.png")
img_1 = cv2.imread("affine_2.png")

#extract feature points
kp1, des1 = detect.extractSURFFeatures(img)
kp2, des2 = detect.extractSURFFeatures(img_1)

#save distorted image to default directory, i.e., distorted_img
distort.saveDistortedImages(img)

#match feature points
matches = match.matchFeature(des1, kp1, des2, kp2)
# Draw matches.
#img3 = cv2.drawMatchesKnn(img,kp1,img_1,kp2,matches, None, flags=2)
#img3 = cv2.drawMatches(img,kp1,img_1,kp2,matches, None, flags=2)
#
#cv2.imshow("match", img3)
#cv2.waitKey(0)
img3 = match.drawMatches(img,kp1,img_1,kp2,matches, thickness=1)

#t = detect.extractDistinctFeatures(img)
#for i in t:
#    cv2.imshow("distorted images",i)
#    cv2.waitKey(0)

cv2.destroyAllWindows()

#test ORB feature point
kps, des = detect.extractORBFeatures(img)
img2 = cv2.imread("distorted_img/rotate_5.png")
kps2, des2 = detect.extractORBFeatures(img2)

if DEBUG:
    trackFeatureChange(img, 5, 0.1, 0.1, 0.1)
    print(TAG + "feature point number: " + str(len(kps)))
    print(TAG + "rotated image feature point number: " + str(len(kps2)))

img_kp = cv2.drawKeypoints(img, kps, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("image with key points", img_kp)
cv2.waitKey(0)

img_kp = cv2.drawKeypoints(img2, kps2, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("image2 with ORB key points", img_kp)
cv2.waitKey(0)

#debug distinct feature points
kps, des = detect.extractDistinctFeatures(img)

img_kp = cv2.drawKeypoints(img, kps, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("image with distinct key points", img_kp)
cv2.waitKey(0)

