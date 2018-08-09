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
#    m2 = bf.knnMatch(des2, des1, k=2)
    
    # Sort them in the order of their distance.
    #matches = sorted(matches, key = lambda x:x.distance)

    if DEBUG > 0:
        print(TAG + "matchFeature: number of original matches is " + str(len(m1)))
    #ratio test
    m1 = ratioTest(m1)
#    m2 = ratioTest(m2)

    r = [m[0] for m in m1]
    if DEBUG > 0:
        print(TAG + "matchFeature: number of returned matches is " + str(len(r)))

    return r

#if the two best matches are relatively close in distance,
#then there exists a possibility that we make an error if we select one or the other.
def ratioTest(m, threshold=0.7):
    r = []

    for a,b in m:
        if a.distance < threshold * b.distance:
            r.append([a])
    
    return r

def drawMatches(img1, kp1, img2, kp2, matches, thickness = 1, color=None, show_center=False): 
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

    if show_center and len(matches) > 0:
        pt1 = [kp1[m.queryIdx].pt for m in matches]
        pt2 = [np.round(kp2[m.trainIdx].pt).astype(int)+np.array([img1.shape[1],0]) for m in matches]
        mean_x1 = sum([p[0] for p in pt1]) / len(matches)
        mean_y1 = sum([p[1] for p in pt1]) / len(matches)
        mean_x2 = sum([p[0] for p in pt2]) / len(matches)
        mean_y2 = sum([p[1] for p in pt2]) / len(matches)
        cv2.circle(new_img, tuple(np.round((mean_x1, mean_y1)).astype(int)), r, (255,255,255), thickness)
        cv2.circle(new_img, tuple(np.round((mean_x2, mean_y2)).astype(int)), r, (255,255,255), thickness)
    
    plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    plt.show()

"""return center of a list of key points

    Args:
        kps: a list of key points
"""
def getCenter(kps):
    pts = [k.pt for k in kps]
    center = (sum([pt[0] for pt in pts])/len(pt), sum([pt[1] for pt in pts])/len(pts))

    return center

"""return a list of sub-lists which contain k neighbor peers in four sub-fields of each key point, i.e., left, right, up, down. Key points in each field are stored in one list

    Args:
        kps:        a list of KeyPoints
        n:          the number of logged closest feature points in each sub-field

    Return:
        [[[left neighbors], [right neighbors], [up neighbors], [down neighbors]],[[...],[...],[...],[...]]...]
"""
def findNeighbors(kps, n=10):
    def getSquareDistance(pt1, pt2):
        return (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2
    r = []
    for k in kps:
        dist= [getSquareDistance(k.pt, kp.pt) for kp in kps]
        #right
        right = []
        #left
        left = []
        #up
        up = []
        #down
        down = []
        for i in range(0,len(kps)):
            d = kps[i]
            if d == k:
                continue
            if d.pt[0] > k.pt[0]:
                right.append(i)
            else:
                left.append(i)
            if (d.pt[1] < k.pt[1]):
                up.append(i)
            else:
                down.append(i)
        sorted(left, key = lambda x:dist[x])
        sorted(right, key = lambda x:dist[x])
        sorted(up, key = lambda x:dist[x])
        sorted(down, key = lambda x:dist[x])
        if DEBUG > 1:
            print(TAG + "point position: " + str([kp.pt for kp in kps]))
            print(TAG + "point distance: " + str(dist))
            print(TAG + "left: " + str(left))
            print(TAG + "right: " + str(right))
            print(TAG + "up: " + str(up))
            print(TAG + "down: " + str(down))

        r.append([left[:n], right[:n], up[:n], down[:n]])

    return r


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

"""return confidence with consideration to excluding template feature points that are probably blocked

    Args:
        matches: matches
        query_kps:  query key points, corresponding to match,queryIdx
        template_kps: template key points, corresponding to match,trainIdx
        neighbor_num: the number of neighbors which are taken into consideration
        h_angle: change on horizontal angle. Facing the scenery, the camera holder turning rightward around the scenery is defined as postive angle, and turning leftward as negtive angle.
        v_angle: change on vertical angle. Facing the scenery, the camera holder turning downward around the scenery is defined as postive angle, and turning upward as negtive angle.
        blocked_threshold: threshold used to judge if an unmatched feature point is blocked
"""
def getAdjustedConfidenceByShrinkTemplate(matches, query_kps, template_kps, neighbor_num=10, h_angle=0, v_angle=0, blocked_threshold=0.8):
    #just record the indexes
    matched_kps = [m.trainIdx for m in matches]
    unmatched_kps = [i for i in range(len(template_kps)) if i not in matched_kps]
    if DEBUG > 1:
        print(TAG + "unmatched feature points: " + str(unmatched_kps))

    neighbors = findNeighbors(template_kps, neighbor_num)

    #which part of neighbors should be checked. 0, left, 1, right, 2, up, 3, down
    if h_angle > 0:
        check_index = 0
    elif h_angle < 0:
        check_index = 1
    elif v_angle > 0:
        check_index = 2
    elif h_angle < 0:
        check_index = 3
    else:
        return len(matches)/len(template_kps)

    #record unmatched feature points that are probably blocked
    blocked = 0
    for i in unmatched_kps:
        nbs = neighbors[i][check_index]
        if DEBUG > 1:
            print("%s neighbor points of %s: %s" % (TAG, str(template_kps[i].pt), str(nbs)))
        if len(nbs) == 0:
            continue
        matched_nb = [nb for nb in nbs if nb in matched_kps]
        if len(matched_nb)/len(nbs) > blocked_threshold:
            blocked += 1
    if DEBUG > 0:
        print(TAG + "probably blocked feature points: " + str(blocked))
    return len(matches)/(len(template_kps)-blocked)
