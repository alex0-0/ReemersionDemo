#!/usr/bin/env python3
""" Match feature points """
import cv2

def matchFeature(des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher()
#    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    
    # Match descriptors.
    m1 = bf.knnMatch(des1, des2, k=2)
    m2 = bf.knnMatch(des2, des1, k=2)
    
    # Sort them in the order of their distance.
    #matches = sorted(matches, key = lambda x:x.distance)

    #ratio test
    m1 = ratioTest(m1)
    m2 = ratioTest(m2)

    #symmetry test
    sym_match = symmetryTest(m1, m2)

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
    
def ransacTest(m, k1, k2):
    r = []
