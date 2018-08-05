#!/usr/bin/env python3
""" Match feature points """
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

#0: turn off debug mode
#1: print out necessary debug log
#2: print out verbose log
DEBUG = 1   
TAG = "MATCH\t"

class DescriptorType(Enum):
    ORB = 1
    SURF = 2

def BFMatchFeature(des1, des2, d_type=DescriptorType.ORB):
    if d_type == DescriptorType.SURF:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    if DEBUG > 0:
        print(TAG + "query key points size: " + str(len(des1)))
        print(TAG + "train key points size: " + str(len(des2)))

    m = bf.match(des1, des2)
    return m

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

    if DEBUG > 0:
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

"""return estimated origin of keypoints
h_angle stands for horizontal angle change
v_angle stands for vertical angle change
"""
def getCenter(kps, h_angle, v_angle):
    a=0

"""return average square distance from points to center and 
   a list containing square distance from points to center

   Args:
    pts: points, a list containing the position of every points
    center: center point, (x,y)
"""
def getAverageSquareDistance(pts, center):
    (cx, cy) = center
    r = [(x-cx)**2 + (y-cy)**2 for (x,y) in pts]
    average = sum(r) / len(pts)
    if DEBUG > 1:
        print(TAG + "average square distance: " + str(average))
        print(TAG + "square distance: " + str(r))

    return average, r

def decideWeithsBySquareDistance(center, kps):
    pos = [k.pt for k in kps]
    if DEBUG > 1:
        print(TAG + "key points position: " + str(pos))
    ave, dis = getAverageSquareDistance(pos, center)
    return [d/ave for d in dis]


def assignWeights(center, kps):
    return decideWeithsBySquareDistance(center, kps)

"""return changed normalized center( ps: for now we haven't yet consider rotation change brings difference to key points!!!)

    Args:
        ori_center: normalized original center, e.g., (image_width/2, image_height/2) is (0.5, 0.5)
        h_angle: change on horizontal angle. Facing the scenery, the camera holder turning rightward around the scenery is defined as postive angle, and turning leftward as negtive angle.
        v_angle: change on vertical angle. Facing the scenery, the camera holder turning upward around the scenery is defined as postive angle, and turning downward as negtive angle.
"""
def decideCenterByOrientation(ori_center, h_angle, v_angle):
    (x, y) = ori_center 
    if h_angle < 90 and h_angle > -90:
        x = x * (1 - h_angle/90)
    if v_angle < 90 and v_angle > -90:
        y = y * (1 - h_angle/90)
    x = 1 if x > 1 else x
    y = 1 if y > 1 else y
    return (x, y)

def getWeightedMatchingConfidence(img_width, img_height, matches, h_angle, v_angle, template_kps):
    (x, y) = decideCenterByOrientation((1/2, 1/2), h_angle, v_angle)
    x = x * img_width
    y = y * img_height

    #get weights for every template keypoints
    weights = assignWeights((x,y), template_kps)

    if DEBUG > 1:
        print(TAG + "weights: " + str(weights))

    scores = [weights[m.trainIdx] for m in matches]

    return (sum(scores) / len(template_kps))
