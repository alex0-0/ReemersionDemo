#!/usr/bin/env python3
""" Test functions to work """
import cv2
import detect
import numpy as np
import distort
import match

DEBUG = True
TAG = "TEST\t"

def trackFeatureChange(img, angle_step, scale_step, affine_step, pers_step, detect_method=detect.extractORBFeatures):
    kp1, des1 = detect_method(img)
    if detect_method == detect.extractSURFFeatures:
        if DEBUG:
            print(TAG + "SURF descriptor")
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    else:
        if DEBUG:
            print(TAG + "ORB descriptor")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    print(TAG + "original: " + str(len(kp1)))

    print(TAG + "**************rotate**************")
    t = distort.rotateImage(img)
    count = 0
    for i in t:
        kp, des = detect_method(i)
        print(TAG + ("+" if (count%2==0) else "-") + str((count//2+1)*angle_step) + ": " + str(len(kp)))
        m = bf.match(des1, des)
        print(TAG + "matched features: " + str(len(m)))
        count += 1

    print(TAG + "**************scale**************")
    t = distort.scaleImage(img)
    count = 0
    for i in t:
        kp, des = detect_method(i)
        scale = (1 + (count//2+1)*scale_step) if (count%2==0) else (1 - (count//2+1)*scale_step)
        print(TAG + str(scale) + ": " + str(len(kp)))
        m = bf.match(des1, des)
        print(TAG + "matched features: " + str(len(m)))
        count += 1

    print(TAG + "**********scale + rotate**********")
    t = distort.scaleImage(img)
    count = 0
    for i in t:
        scale = (1 + (count//2+1)*scale_step) if (count%2==0) else (1 - (count//2+1)*scale_step)
        print(TAG + str(scale) + ": ")
        im = distort.rotateImage(i)
        c = 0
        for k in im:
            kp, des = detect_method(k)
            print(TAG + ("+" if (c%2==0) else "-") + str((c//2+1)*angle_step) + ": " + str(len(kp)))
            m = bf.match(des1, des)
            print(TAG + "matched features: " + str(len(m)))
            c += 1
        count += 1

    print(TAG + "**************affine**************")
    t = distort.affineImage(img)
    count = 1
    for i in t:
        kp, des = detect_method(i)
        print(TAG + str(count) + ": " + str(len(kp)))
        m = bf.match(des1, des)
        print(TAG + "matched features: " + str(len(m)))
        count += 1

    print(TAG + "perspective:")
    t = distort.changeImagePerspective(img)
    count = 1
    for i in t:
        kp, des = detect_method(i)
        print(TAG + str(count) + ": " + str(len(kp)))
        m = bf.match(des1, des)
        print(TAG + "matched features: " + str(len(m)))
        count += 1

def testDetect(img, detect_method=detect.extractORBFeatures, title=None):
    kps, des = detect_method(img)
    img_kp = cv2.drawKeypoints(img, kps, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #display detected feature points
    t = "image with key points"
    if title != None:
        t = title
    cv2.imshow(t, img_kp)
    cv2.waitKey(0)

def testDistinctFeature(img, detect_method=detect.extractORBFeatures):
    kps, des = detect.extractDistinctFeatures(img)
    if DEBUG:
        print(TAG + "distinct feature points: " + str(len(kps)))
    img_kp = cv2.drawKeypoints(img, kps, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("image with distinct key points", img_kp)
    cv2.waitKey(0)

def testMatch(img_1, img_2, detect_method=detect.extractORBFeatures):
    #extract feature points
    kp1, des1 = detect_method(img_1)
    kp2, des2 = detect_method(img_2)
    #match feature points
    matches = match.matchFeature(des1, kp1, des2, kp2)
    # Draw matches.
    #img3 = cv2.drawMatchesKnn(img,kp1,img_1,kp2,matches, None, flags=2)
    #img3 = cv2.drawMatches(img,kp1,img_1,kp2,matches, None, flags=2)
    #
    #cv2.imshow("match", img3)
    #cv2.waitKey(0)
    if DEBUG:
        print(TAG + "matches size: " + str(len(matches)))
        print(TAG + "query key points size: " + str(len(kp1)))
        print(TAG + "train key points size: " + str(len(kp2)))
    
    match.drawMatches(img_1,kp1,img_2,kp2,matches[:10], thickness=3, color=(255,0,0))
