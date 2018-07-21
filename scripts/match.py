#!/usr/bin/env python3
""" Match feature points """
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

DEBUG = False
TAG = "MATCH\t"

class DescriptorType(Enum):
    ORB = 1
    SURF = 2

def matchFeature(des1, kp1, des2, kp2, d_type=DescriptorType.ORB):
    # create BFMatcher object
    '''
    By default, BFMatcher uses cv2.NORM_L2. It is good for SIFT, SURF etc (cv2.NORM_L1 is also there). For binary string based descriptors like ORB, BRIEF, BRISK etc, cv2.NORM_HAMMING should be used, which used Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used.
    '''
    def distanceType(x):
        return {
                DescriptorType.ORB : cv2.NORM_HAMMING,
                DescriptorType.SURF : cv2.NORM_L2
                }.get(x, cv2.NORM_HAMMING)

    if DEBUG:
        print(TAG + "descriptor type: " + str(d_type))

    bf = cv2.BFMatcher(distanceType(d_type))
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

    sym_match = sorted(sym_match, key = lambda x:x.distance)

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

def drawMatches(img1, kp1, img2, kp2, matches, thickness = 1, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.

    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.

    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    if DEBUG:
        print(TAG + "matches size: " + str(len(matches)))
        print(TAG + "query key points size: " + str(len(kp1)))
        print(TAG + "train key points size: " + str(len(kp2)))
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    plt.show()

