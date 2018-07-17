#!/usr/bin/env python3
""" Match feature points """
import cv2
import numpy as np

def matchFeature(des1, kp1, des2, kp2):
    # create BFMatcher object
    bf = cv2.BFMatcher()
#    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    
    # Match descriptors.
    #cv.DescriptorMatcher.knnMatch(	queryDescriptors, trainDescriptors, k[, mask[, compactResult]]	)
    m1 = bf.knnMatch(des1, des2, k=2)
    m2 = bf.knnMatch(des2, des1, k=2)
    
    # Sort them in the order of their distance.
    #matches = sorted(matches, key = lambda x:x.distance)

    #ratio test
    m1 = ratioTest(m1)
    m2 = ratioTest(m2)

    #symmetry test
    sym_match = symmetryTest(m1, m2)
    
    kRansacThreshold = 20
    if len(sym_match) > kRansacThreshold:
        sym_match = ransacTest(sym_match, kp1, kp2)

    return sym_match

#if the two best matches are relatively close in distance,
#then there exists a possibility that we make an error if we select one or the other.
def ratioTest(m):
    #threshold
    kRatio = 0.7
    r = []

    for a,b in m:
        if a.distance < kRatio * b.distance:
            r.append([a])
    
    return r

def symmetryTest(m1, m2):
    r = []
    for match_1 in m1:
        for match_2 in m2:
            if match_1[0].queryIdx == match_2[0].trainIdx and match_2[0].queryIdx == match_1[0].trainIdx:
                r.append(match_1[0])
                break

    return r
    
def ransacTest(matches, kp1, kp2):
    r = []
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    kRansacRepojThreashold = 15.0
    #RANSAC get mask
    T, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, kRansacRepojThreashold)

    #filter out inlier
    i = 0
    for m in matches:
        if 1 == mask[i]:
            r.append(m)
        i += 1

    return r

