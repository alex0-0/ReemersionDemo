#!/usr/bin/env python3
""" Detect feature points """
import cv2
import match
from match import DescriptorType

DEBUG = False
TAG = "DETECT\t"

def extractSURFFeatures(img):
    surf = cv2.xfeatures2d.SURF_create()
    #TODO: need to figure out how to increase the threshold
    #surf.hessianThreshold = 400
    keypoints, descriptors = surf.detectAndCompute(img, None)
    return keypoints, descriptors

def extractORBFeatures(img):
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=90, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
    
    kp = orb.detect(img)
    kp, des = orb.compute(img, kp)

    return kp,des

