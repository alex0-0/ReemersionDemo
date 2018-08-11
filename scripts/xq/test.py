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
DEBUG = 0   
TAG = "TEST\t"

def testDetect(img, detect_method=detect.extractORBFeatures, title=None):
    kps, des = detect_method(img)
    img_kp = cv2.drawKeypoints(img, kps, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #display detected feature points
    t = "image with key points"
    if title != None:
        t = title
    cv2.namedWindow(t, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(t,600,600)
    cv2.imshow(t, img_kp)
    cv2.waitKey(0)

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
    
    match.drawMatches(img_1,kp1,img_2,kp2,matches, thickness=3, color=(255,0,0))

def findMatches(query_img,template_img,detect_method=detect.extractORBFeatures):
    kp1, des1 = detect_method(query_img)
    kp2, des2 = detect_method(template_img)
    #match feature points
    #cv.DescriptorMatcher.knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]])
    if detect_method == detect.extractORBFeatures:
        matches = match.BFMatchFeature(des1, des2, DescriptorType.ORB)
    elif detect_method == detect.extractSURFFeatures:
        matches = match.BFMatchFeature(des1, des2, DescriptorType.SURF)
    return matches

def filterFP(matches, distance_threshold):
    matches = sorted(matches, key=lambda m:m.distance)
    filtered_matches = [m for m in matches if m.distance<distance_threshold]
    return filtered_matches

"""
    test how distance threshold and ratio threshold influence the matches

    Args:
        img1: query image
        img2: template image
        distance_threshold: filter out matches whose distance is larger than the threshold
        ratio_threshold: filter out matches whose best match's distance is larger than the second best match's distance*threshold
        detect_method: detect method used for finding feature points
        show_image: decide if the result image should be presented or not
        matches_display_num: how many matches should be displayed on image, 0 and negative number stand for showing all
"""
def testMatchWithDistanceAndRatio(img1, img2, detect_method=detect.extractORBFeatures, distance_threshold=50, ratio_threshold=0.7, show_image=False, matches_display_num=0):
    kp1, des1 = detect_method(img1)
    kp2, des2 = detect_method(img2)
    if detect_method == detect.extractORBFeatures:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    elif detect_method == detect.extractSURFFeatures:
        bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    matches = [m for m in matches if m is not None and m[0].distance < distance_threshold]
    matches = match.ratioTest(matches, ratio_threshold)
    matches = [m[0] for m in matches]
    #display 10 best matches
    if show_image:
        m = matches
        if matches_display_num > 0:
            matches = sorted(matches, key=lambda m:m.distance)
            m = matches[:matches_display_num]
        match.drawMatches(img1,kp1,img2,kp2,m, thickness=2, color=(255,0,0), show_center=True)
    return matches


"""
    test how the weight assignment work

    Args:
        query_img: query image
        template_img: template image
        h_angle: horizontal angle change
        v_angle: vertical angle change
        distance_threshold: filter out matches whose distance is larger than the threshold
        detect_method: detect method used for finding feature points
        show_image: decide if the result image should be presented or not
        matches_display_num: how many matches should be displayed on image
"""
def testweightedmatching(query_img, template_img, h_angle, v_angle, distance_threshold=50, detect_method=detect.extractORBFeatures, show_image=False, matches_display_num=0):
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
    if DEBUG > 0:
        print(TAG + "average distance: " + str(sum([m.distance for m in filtered_matches])/len(filtered_matches)))

    #display 10 best matches
    if show_image:
        m = matches
        if matches_display_num > 0:
            m = matches[:matches_display_num]
        match.drawMatches(query_img,kp1,template_img,kp2,m, thickness=2, color=(255,0,0), show_center=True)

    #assume two images have same size
    height, width = template_img.shape[:2]
    score = match.getWeightedMatchingConfidence(width, height, filtered_matches, h_angle, v_angle, kp2)

    if DEBUG > 0:
        print(TAG + "template feature points: " + str(len(des2)))
        print(TAG + "matched points: " + str(len(matches)))
        print(TAG + "precision: " + str(len(matches)/len(des2)))
        print(TAG + "threshold: " + str(distance_threshold) + "\tfiltered matched points: " + str(len(filtered_matches)))

    print(TAG + "testWeightedMatching: weighted score is " + str(score))

def testFindNeighbors(img, neighboring_num=10, detect_method=detect.extractORBFeatures):
    kp,des = detect_method(img)
    neighbor_points = match.findNeighbors(kp, neighboring_num)
    for i in range(len(kp)):
        print(TAG + "feature point: " + str(kp[i].pt))
        print(TAG + "left: " + str([kp[np].pt for np in neighbor_points[i][0]]))
        print(TAG + "right: " + str([kp[np].pt for np in neighbor_points[i][1]]))
        print(TAG + "up: " + str([kp[np].pt for np in neighbor_points[i][2]]))
        print(TAG + "down: " + str([kp[np].pt for np in neighbor_points[i][3]]))

"""
    test how the adjusted confidence work

    Args:
        query_img: query image
        template_img: template image
        h_angle: horizontal angle change
        v_angle: vertical angle change
        distance_threshold: filter out matches whose distance is larger than the threshold
        detect_method: detect method used for finding feature points
        show_image: decide if the result image should be presented or not
        matches_display_num: how many matches should be displayed on image
"""
def testAdjustedConfidence(query_img, template_img, h_angle=0, v_angle=0, distance_threshold=100, blocked_threshold=0.8, neighbor_num=10, detect_method=detect.extractORBFeatures, show_image=False, matches_display_num=0):
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
        m = matches
        if matches_display_num > 0:
            m = matches[:matches_display_num]
        match.drawMatches(query_img,kp1,template_img,kp2,m, thickness=2, color=(255,0,0), show_center=True)

    #assume two images have same size
    height, width = template_img.shape[:2]
    score, blocked= match.getAdjustedConfidenceByShrinkTemplate(filtered_matches, kp1, kp2, neighbor_num=neighbor_num, h_angle=h_angle, v_angle=v_angle, blocked_threshold=blocked_threshold)

    if DEBUG > 0:
        print(TAG + "template feature points: " + str(len(des2)))
        print(TAG + "matched points: " + str(len(matches)))
        print(TAG + "distance threshold: " + str(distance_threshold) + "\tfiltered matched points: " + str(len(filtered_matches)))
        print(TAG + "precision: " + str(len(filtered_matches)/len(des2)))
        print(TAG + "testAdjustedConfidence: adjusted score is " + str(score))
    return score, blocked
