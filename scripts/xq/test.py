#!/usr/bin/env python3
""" Test functions to work """
import cv2
import detect
import numpy as np
import match
from match import DescriptorType
import os
import matplotlib.pyplot as plt
import glob

#0: turn off debug mode
#1: print out necessary debug log
#2: print out verbose log
DEBUG = 1   
TAG = "TEST\t"

def testMatch(img_1, img_2, detect_method=detect.extractORBFeatures):
    #extract feature points
    kp1, des1 = detect_method(img_1)
    kp2, des2 = detect_method(img_2)
    #match feature points
    if detect_method == detect.extractORBFeatures:
        matches = match.matchFeature(des1, kp1, des2, kp2, DescriptorType.ORB)
    elif detect_method == detect.extractSURFFeatures:
        matches = match.matchFeature(des1, kp1, des2, kp2, DescriptorType.SURF)
    # Draw matches.
    #img3 = cv2.drawMatchesKnn(img,kp1,img_1,kp2,matches, None, flags=2)
    #img3 = cv2.drawMatches(img,kp1,img_1,kp2,matches, None, flags=2)
    #
    #cv2.imshow("match", img3)
    #cv2.waitKey(0)
    if DEBUG > 0:
        print(TAG + "matches size: " + str(len(matches)))
        print(TAG + "query key points size: " + str(len(kp1)))
        print(TAG + "train key points size: " + str(len(kp2)))
    
    match.drawMatches(img_1,kp1,img_2,kp2,matches[:10], thickness=3, color=(255,0,0))

def testWeightedMatching(query_img, template_img, h_angle, v_angle, distance_threshold=50, detect_method=detect.extractORBFeatures, show_image=False):
    #extract feature points
    kp1, des1 = detect_method(query_img)
    kp2, des2 = detect_method(template_img)
    #match feature points
    #cv.DescriptorMatcher.knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]])
    if detect_method == detect.extractORBFeatures:
        matches = match.BFMatchFeature(des1, des2, DescriptorType.ORB)
    elif detect_method == detect.extractSURFFeatures:
        matches = match.BFMatchFeature(des1, des2, DescriptorType.SURF)
    
    #arrange matches by its distance
    matches = sorted(matches, key=lambda m:m.distance)
    filtered_matches = [m for m in matches if m.distance<distance_threshold]

    #display 10 best matches
    if show_image:
        match.drawMatches(query_img,kp1,template_img,kp2,matches[:10], thickness=3, color=(255,0,0))

    #assume two images have same size
    height, width = template_img.shape[:2]
    score = match.getWeightedMatchingConfidence(width, height, filtered_matches, h_angle, v_angle, kp2)

    if DEBUG > 0:
        print(TAG + "template feature points: " + str(len(des2)))
        print(TAG + "matched points: " + str(len(matches)))
        print(TAG + "precision: " + str(len(matches)/len(des2)))
        print(TAG + "threshold: " + str(distance_threshold) + "\tfiltered matched points: " + str(len(filtered_matches)))

    print(TAG + "testWeightedMatching: weighted score is " + str(score))
