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

'''
Parameters:
    Parameters:
nfeatures – The maximum number of features to retain.
scaleFactor – Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.
nlevels – The number of pyramid levels. The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, nlevels).
edgeThreshold – This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
firstLevel – It should be 0 in the current implementation.
WTA_K – The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
scoreType – The default HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to KeyPoint::score and is used to retain best nfeatures features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.
patchSize – size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.
'''
def extractORBFeatures(img):
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
    
    #kp[0].pt is the postion of key point in the image
    kp = orb.detect(img)
    kp, des = orb.compute(img, kp)

    return kp,des

def extractSIFTFeatures(img):
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, des) = sift.detectAndCompute(img, None)
    return kps,des
