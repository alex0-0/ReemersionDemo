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
import argparse
import sys

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

def testMatchPrecision(img_1, img_2, detect_method=detect.extractORBFeatures):
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
    return len(matches)/len(kp2)
    

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
    
    height, width = img.shape[:2]
    idx=1
    
    for i in range(len(kp)):
        if not i==idx:
            continue
        print(TAG + "feature point: " + str(kp[i].pt))
        print(TAG + "left: " +str(len(neighbor_points[i][0]))+"--"+ str([kp[np].pt for np in neighbor_points[i][0]]))
        print(TAG + "right: " + str(len(neighbor_points[i][1]))+"--"+str([kp[np].pt for np in neighbor_points[i][1]]))
        print(TAG + "up: " + str(len(neighbor_points[i][2]))+"--"+str([kp[np].pt for np in neighbor_points[i][2]]))
        print(TAG + "down: " + str(len(neighbor_points[i][3]))+"--"+str([kp[np].pt for np in neighbor_points[i][3]]))


    cv2.circle(img,tuple(np.round(kp[idx].pt).astype(int)), 20, (0,255,255), -1)

    c=[(255,0,0), (0,0,255), (0,255,0), (100,100,100)]

    cv2.circle(img,(100,int(round(height/2))), 50, c[0], -1)
    cv2.circle(img,(width-100,int(round(height/2))), 50, c[1], -1)
    cv2.circle(img,(int(round(width/2)),100), 50, c[2], -1)
    cv2.circle(img,(int(round(width/2)),height-100), 50, c[3], -1)

    for i in [0,1,2,3]:
        nbs=neighbor_points[idx][i]
        for nb in nbs:
            cv2.circle(img,tuple(np.round(kp[nb].pt).astype(int)), 10, c[i], -1)


#t = "image with neighbor points"
#   cv2.namedWindow(t, cv2.WINDOW_NORMAL)
#   cv2.resizeWindow(t,600,600)
#   cv2.imshow(t, img)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
    plt.imshow(img)
    plt.show()
    return neighbor_points

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
def testAdjustedConfidence(query_img, template_img, h_angle=0, v_angle=0, distance_threshold=0.3, blocked_threshold=0.8, neighbor_num=10, detect_method=detect.extractORBFeatures, show_image=False, matches_display_num=0, pos_dis=100):
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
#    filtered_matches = [m for m in matches if m.distance<distance_threshold]
    filtered_matches = matches[:int(distance_threshold*len(kp2))]

    #assume two images have same size
    height, width = template_img.shape[:2]
    score, tp, blocked, nbs = match.getAdjustedConfidenceByShrinkTemplateNew(filtered_matches, kp1, kp2, neighbor_num=neighbor_num, h_angle=h_angle, v_angle=v_angle, blocked_threshold=blocked_threshold, dis_threshold=pos_dis, return_neighbors=True)

    #display matches_display_num best matches
    if show_image:
        m = filtered_matches
        if matches_display_num > 0:
            m = filtered_matches[:matches_display_num]
        #just pick a few neighbors for debug
        bfp=2
        neighbor_of_blocked = set()
        [[neighbor_of_blocked.add(kp2[i]) for i in nb] for nb in nbs[:bfp]]
#print(TAG+str(len(neighbor_of_blocked)))

        match.drawMatches(query_img,kp1,template_img,kp2,m, thickness=2, color=(255,0,0), show_center=True, custom_point1=[kp2[i] for i in blocked[:bfp]], custom_point2=neighbor_of_blocked)

    if DEBUG > 0:
        print(TAG + "template feature points: " + str(len(des2)))
        print(TAG + "matched points: " + str(len(matches)))
        print(TAG + "distance threshold: " + str(distance_threshold) + "\tfiltered matched points: " + str(len(filtered_matches)))
        print(TAG + "precision: " + str(len(filtered_matches)/len(des2)))
        print(TAG + "testAdjustedConfidence: adjusted score is " + str(score))
    return score, len(blocked), tp

def batchTest(directory, blocked_threshold, distance, pos_dis, neighbor_num, detect_method=detect.extractORBFeatures):
    def readImage(f):
        return cv2.imread(directory+"/"+f)
    
    img000 = readImage("000.JPG")
    imgP15 = readImage("015.JPG")
    imgN15 = readImage("-15.JPG")
    imgP30 = readImage("030.JPG")
    imgN30 = readImage("-30.JPG")
    imgP45 = readImage("045.JPG")
    imgN45 = readImage("-45.JPG")
    img180 = readImage("180.JPG")
    imgFalse = readImage("false.JPG")

    m000_P15=findMatches(imgP15,img000);
    m000_N15=findMatches(imgN15,img000);
    m000_P30=findMatches(imgP30,img000);
    m000_N30=findMatches(imgN30,img000);
    m000_P45=findMatches(imgP45,img000);
    m000_N45=findMatches(imgN45,img000);
    m000_180=findMatches(img180,img000);
    m000_false=findMatches(img180,imgFalse);
    
    print("Distance filtered matches")
    print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","false"))
    ##test different threshold
#    for d in range(20, 101, 10):
#        fm000_P15=filterFP(m000_P15,d);
#        fm000_N15=filterFP(m000_N15,d);
#        fm000_P30=filterFP(m000_P30,d);
#        fm000_N30=filterFP(m000_N30,d);
#        fm000_P45=filterFP(m000_P45,d);
#        fm000_N45=filterFP(m000_N45,d);
#        fm000_180=filterFP(m000_180,d);
#        fm000_false=filterFP(m000_false,d);

    kps,des = detect_method(img000)
    for d in np.arange(0.2, 1.1, 0.1):
        fm000_P15=m000_P15[:int(d*len(kps))];
        fm000_N15=m000_N15[:int(d*len(kps))];
        fm000_P30=m000_P30[:int(d*len(kps))];
        fm000_N30=m000_N30[:int(d*len(kps))];
        fm000_P45=m000_P45[:int(d*len(kps))];
        fm000_N45=m000_N45[:int(d*len(kps))];
        fm000_180=m000_180[:int(d*len(kps))];
        fm000_false=m000_false[:int(d*len(kps))];
        print("%.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % (d, len(fm000_P15), len(fm000_N15), len(fm000_P30), len(fm000_N30),len(fm000_P45), len(fm000_N45), len(fm000_180),len(fm000_false)))
        
    si = False  #show_image
#    bt = blocked_threshold    #blocked_threshold
#    nn = neighbor_num     #neighbor_number
#    dis = distance
    print("\nAdjusted confidence test number of matches(ratio=%.2f, neighbor_number=%d, pos_dis=%d)"%(blocked_threshold, neighbor_num, pos_dis))
    print("%-5s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","False"))
    #test different threshold
    for d in np.arange(0.2, 1.1, 0.1):
        con000_P15=testAdjustedConfidence(imgP15, img000, blocked_threshold=blocked_threshold, distance_threshold=d, pos_dis=pos_dis, h_angle=15, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_N15=testAdjustedConfidence(imgN15, img000, blocked_threshold=blocked_threshold, distance_threshold=d, pos_dis=pos_dis, h_angle=-15, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_P30=testAdjustedConfidence(imgP30, img000, blocked_threshold=blocked_threshold, distance_threshold=d, pos_dis=pos_dis, h_angle=30, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_N30=testAdjustedConfidence(imgN30, img000, blocked_threshold=blocked_threshold, distance_threshold=d, pos_dis=pos_dis, h_angle=-30, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_P45=testAdjustedConfidence(imgP45, img000, blocked_threshold=blocked_threshold, distance_threshold=d, pos_dis=pos_dis, h_angle=30, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_N45=testAdjustedConfidence(imgN45, img000, blocked_threshold=blocked_threshold, distance_threshold=d, pos_dis=pos_dis, h_angle=-30, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_180=testAdjustedConfidence(img180, img000, blocked_threshold=blocked_threshold, distance_threshold=d, pos_dis=pos_dis, h_angle=180, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_false=testAdjustedConfidence(imgFalse, img000, distance_threshold=d, pos_dis=pos_dis, h_angle=0, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=neighbor_num);
        print("%.2f\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d" % (d, con000_P15[0],con000_P15[2],con000_P15[1], con000_N15[0],con000_N15[2],con000_N15[1], con000_P30[0],con000_P30[2],con000_P30[1], con000_N30[0],con000_N30[2],con000_N30[1],con000_P45[0],con000_P45[2],con000_P45[1], con000_N45[0],con000_N45[2],con000_N45[1], con000_180[0],con000_180[2],con000_180[1],con000_false[0],con000_false[2],con000_false[1]))

    print("\nAdjusted confidence test ratio(distance=%.2f, neighbor_number=%d, pos_dis=%d)"%(distance,neighbor_num, pos_dis))
    print("%-5s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","False"))
    #test different threshold
    for bt in np.arange(0.2, 1.1, 0.1):
        con000_P15=testAdjustedConfidence(imgP15, img000, neighbor_num=neighbor_num, distance_threshold=distance, pos_dis=pos_dis, h_angle=15, show_image=si, matches_display_num=100, blocked_threshold=bt);
        con000_N15=testAdjustedConfidence(imgN15, img000, neighbor_num=neighbor_num, distance_threshold=distance, pos_dis=pos_dis, h_angle=-15, show_image=si, matches_display_num=100, blocked_threshold=bt);
        con000_P30=testAdjustedConfidence(imgP30, img000, neighbor_num=neighbor_num, distance_threshold=distance, pos_dis=pos_dis, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=bt);
        con000_N30=testAdjustedConfidence(imgN30, img000, neighbor_num=neighbor_num, distance_threshold=distance, pos_dis=pos_dis, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=bt);
        con000_P45=testAdjustedConfidence(imgP45, img000, neighbor_num=neighbor_num, distance_threshold=distance, pos_dis=pos_dis, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=bt);
        con000_N45=testAdjustedConfidence(imgN45, img000, neighbor_num=neighbor_num, distance_threshold=distance, pos_dis=pos_dis, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=bt);
        con000_180=testAdjustedConfidence(img180, img000, neighbor_num=neighbor_num, distance_threshold=distance, pos_dis=pos_dis, h_angle=180, show_image=si, matches_display_num=100, blocked_threshold=bt);
        con000_false=testAdjustedConfidence(imgFalse, img000, neighbor_num=neighbor_num, distance_threshold=distance, pos_dis=pos_dis, h_angle=0, show_image=si, matches_display_num=100, blocked_threshold=bt);
    #print("%.2f\t%d\t%d\t%d\t%d\t%d\t%d" % (bt, con000_P15[1], con000_N15[1], con000_P30[1], con000_N30[1], con000_180[1],con000_false[1]))
        print("%.02f\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d" % (bt, con000_P15[0],con000_P15[2],con000_P15[1], con000_N15[0],con000_N15[2],con000_N15[1], con000_P30[0],con000_P30[2],con000_P30[1], con000_N30[0],con000_N30[2],con000_N30[1],con000_P45[0],con000_P45[2],con000_P45[1], con000_N45[0],con000_N45[2],con000_N45[1], con000_180[0],con000_180[2],con000_180[1],con000_false[0],con000_false[2],con000_false[1]))

    print("\nAdjusted confidence test neighbor number(ratio=%.2f, distance=%.2f, pos_dis=%d)"%(blocked_threshold,distance, pos_dis))
    print("%-5s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","False"))
    #test different threshold
    for nn in np.arange(4, 11, 1):
        con000_P15=testAdjustedConfidence(imgP15, img000, distance_threshold=distance, pos_dis=pos_dis, h_angle=15, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=nn);
        con000_N15=testAdjustedConfidence(imgN15, img000, distance_threshold=distance, pos_dis=pos_dis, h_angle=-15, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=nn);
        con000_P30=testAdjustedConfidence(imgP30, img000, distance_threshold=distance, pos_dis=pos_dis, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=nn);
        con000_N30=testAdjustedConfidence(imgN30, img000, distance_threshold=distance, pos_dis=pos_dis, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=nn);
        con000_P45=testAdjustedConfidence(imgP45, img000, distance_threshold=distance, pos_dis=pos_dis, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=nn);
        con000_N45=testAdjustedConfidence(imgN45, img000, distance_threshold=distance, pos_dis=pos_dis, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=nn);
        con000_180=testAdjustedConfidence(img180, img000, distance_threshold=distance, pos_dis=pos_dis, h_angle=180, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=nn);
        con000_false=testAdjustedConfidence(imgFalse, img000, distance_threshold=distance, pos_dis=pos_dis, h_angle=0, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=nn);
    #print("%d\t%d\t%d\t%d\t%d\t%d\t%d" % (nn, con000_P15[1], con000_N15[1], con000_P30[1], con000_N30[1], con000_180[1],con000_false[1]))
        print("%d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d" % (nn, con000_P15[0],con000_P15[2],con000_P15[1], con000_N15[0],con000_N15[2],con000_N15[1], con000_P30[0],con000_P30[2],con000_P30[1], con000_N30[0],con000_N30[2],con000_N30[1],con000_P45[0],con000_P45[2],con000_P45[1], con000_N45[0],con000_N45[2],con000_N45[1], con000_180[0],con000_180[2],con000_180[1],con000_false[0],con000_false[2],con000_false[1]))

    print("\nAdjusted confidence test position distance(ratio=%.2f, m_dis=%.2f, neighbor_num=%d)"%(blocked_threshold,distance,neighbor_num))
    print("%-5s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","False"))
    #test different threshold
    for pd in np.arange(50, 151, 10):
        con000_P15=testAdjustedConfidence(imgP15, img000, blocked_threshold=blocked_threshold, distance_threshold=distance, pos_dis=pd, h_angle=15, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_N15=testAdjustedConfidence(imgN15, img000, blocked_threshold=blocked_threshold, distance_threshold=distance, pos_dis=pd, h_angle=-15, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_P30=testAdjustedConfidence(imgP30, img000, blocked_threshold=blocked_threshold, distance_threshold=distance, pos_dis=pd, h_angle=30, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_N30=testAdjustedConfidence(imgN30, img000, blocked_threshold=blocked_threshold, distance_threshold=distance, pos_dis=pd, h_angle=-30, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_P45=testAdjustedConfidence(imgP45, img000, blocked_threshold=blocked_threshold, distance_threshold=distance, pos_dis=pd, h_angle=30, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_N45=testAdjustedConfidence(imgN45, img000, blocked_threshold=blocked_threshold, distance_threshold=distance, pos_dis=pd, h_angle=-30, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_180=testAdjustedConfidence(img180, img000, blocked_threshold=blocked_threshold, distance_threshold=distance, pos_dis=pd, h_angle=180, show_image=si, matches_display_num=100, neighbor_num=neighbor_num);
        con000_false=testAdjustedConfidence(imgFalse, img000, distance_threshold=distance, pos_dis=pd, h_angle=0, show_image=si, matches_display_num=100, blocked_threshold=blocked_threshold, neighbor_num=neighbor_num);
    #print("%d\t%d\t%d\t%d\t%d\t%d\t%d" % (nn, con000_P15[1], con000_N15[1], con000_P30[1], con000_N30[1], con000_180[1],con000_false[1]))
        print("%d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d" % (pd, con000_P15[0],con000_P15[2],con000_P15[1], con000_N15[0],con000_N15[2],con000_N15[1], con000_P30[0],con000_P30[2],con000_P30[1], con000_N30[0],con000_N30[2],con000_N30[1],con000_P45[0],con000_P45[2],con000_P45[1], con000_N45[0],con000_N45[2],con000_N45[1], con000_180[0],con000_180[2],con000_180[1],con000_false[0],con000_false[2],con000_false[1]))


def helpMsg():
    msg = "usage:\n"
    msg += "\tTest!"
    return msg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=helpMsg())
    parser.add_argument("-d", "--directory", help="image directory name, default value is current directory", type=str, required=True, dest="directory")
    parser.add_argument("-o", "--output", help="output file, default value is output", type=str, default="output", dest="output")
    parser.add_argument("-e", "--error", help="error output file, default value is error", type=str, default="error", dest="error")
    parser.add_argument("-md", "--match_distance", help="threshold value for matches number, default value is 0.2", type=float, default=0.2, dest="m_dis")
    parser.add_argument("-pd", "--pos_distance", help="distance threshold value for feature point position, default value is 100", type=int, default=100, dest="p_dis")
    parser.add_argument("-b", "--blocked", help="threshold for judging wheter feature point is blocked, default value is 0.5", type=float, default=0.5, dest="bt")
    parser.add_argument("-n", "--neighbor", help="neighbor number, default value is 10", type=int, default=10, dest="nn")

    args = parser.parse_args()
    output = "%s_%s_md_%d_bt_%.2f_nn_%d_pd_%d" % (args.output, args.directory, args.m_dis, args.bt, args.nn, args.p_dis)
    error = "%s_%s_md_%d_bt_%.2f_nn_%d_pd_%d" % (args.error, args.directory, args.m_dis, args.bt, args.nn, args.p_dis)

    #check if the output file exist, if yes, add a suffix
    if os.path.exists(output):
        i = 0
        while os.path.exists("%s_%s" % (output, i)):
            i += 1
        output = "%s_%s" % (output, i)
    if os.path.exists(error):
        i = 0
        while os.path.exists("%s_%s" % (error, i)):
            i += 1
        error = "%s_%s" % (error, i)
    #redirect output to file
    sys.stdout = open(output,'a')
    sys.stderr = open(error,'a')
    #sys.stdout = sys.__stdout__    #restore to default stdout

    batchTest(args.directory, args.bt, args.m_dis, args.p_dis, args.nn)
