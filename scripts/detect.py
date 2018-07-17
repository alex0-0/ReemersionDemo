#!/usr/bin/env python3
""" Detect feature points """
import cv2
import distort
import match

DEBUG = False
TAG = "DETECT\t"

#return keypoints and descriptors
def extractFeatures(img):
    surf = cv2.xfeatures2d.SURF_create()
    #TODO: need to figure out how to increase the threshold
#    surf.hessianThreshold = 400
    keypoints, descriptors = surf.detectAndCompute(img, None)
    return keypoints, descriptors

def extractDistinctFeatures(img):
    #every feature points has to be matched to at least more than 3 pictures
    kDistinctThreshold = 10

    #original key points of the image
    kps, des = extractFeatures(img)

    #get distorted images
    distorted_img = []
    distorted_img.extend(distort.rotateImage(img))
    distorted_img.extend(distort.scaleImage(img))
    distorted_img.extend(distort.affineImage(img))
    distorted_img.extend(distort.changeImagePerspective(img))
    
    #store key points of every distorted images
    list_of_kps = []
    #store descriptors of every distorted images
    list_of_des = []

    for i in distorted_img:
        k, d = extractFeatures(i)
        list_of_kps.append(k)
        list_of_des.append(d)
    
    if DEBUG:
        print(TAG + "original key point number:" + str(len(kps)))

    #list of counters which record how many times the feature point is matched to distorted images
    counters = [0] * len(kps)

    for i in range(len(distorted_img)):
        matches = match.matchFeature(des, kps, list_of_des[i], list_of_kps[i])
        for m in matches:
            counters[m.queryIdx] += 1

    if DEBUG:
        print(TAG + "counters of key points of matching original image to distorted images:" + str(counters))

    r_kps = []
    r_des = []
    #filter out those indistinct features
    for i in range(len(counters)):
        if counters[i] > kDistinctThreshold:
            r_kps.append(kps[i])
            r_des.append(des[i])

    if DEBUG:
        print(TAG + "distinct key point number:" + str(len(r_kps)))
    
    return r_kps, r_des

#    distorted_features = [extractFeatures(i) for i in distorted_img]

def extractORBFeatures(img):
    orb = cv2.ORB()

    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)

    return kp,des
