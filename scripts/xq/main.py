#!/usr/bin/env python3
""" Call functions to work, program entry """
import cv2
import detect
import numpy as np
import match
import test
from match import DescriptorType

DEBUG = True
TAG = "MAIN\t"

test.testAlgorithmPrecision()
exit()

#img1 = cv2.imread("000.JPG")
#img2 = cv2.imread("090.JPG")
img1 = cv2.imread("-30.JPG")
img2 = cv2.imread("000.JPG")

#test.testAdjustedConfidence(img1, img2, h_angle=-30, v_angle=0, distance_threshold=50, blocked_threshold=0.5, neighbor_num=10, detect_method=detect.extractORBFeatures, show_image=True, matches_display_num=0)

directory = "horse"
#directory = "horse"
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

#test.testDetect(img000,detect_method=detect.extractSIFTFeatures)
test.testDetect(img000,detect_method=detect.extractORBFeatures)
#test.testDetect(img000,detect_method=detect.extractSURFFeatures)
#test.testMatch(img000, imgN30, detect.extractORBFeatures)
#test.testAdjustedConfidence(imgFalse, img000, distance_threshold=1.0, neighbor_num=5, h_angle=15, show_image=True, matches_display_num=100, blocked_threshold=0.5, pos_dis=150);
exit()

#kp1, des1 = detect.extractORBFeatures(img1)
#kp2, des2 = detect.extractORBFeatures(img2)
#matches = match.BFMatchFeature(des1, des2, DescriptorType.ORB)
#print("image 1 feature number: " + str(len(kp1)))
#print("image 2 feature number: " + str(len(kp2)))
#print("matched feature number: " + str(len(matches)))
#match.drawMatches(img1,kp1,img2,kp2,matches, thickness=1, color=(255,0,0))

#test.testWeightedMatching(img1, img2, 60, 0, 50, show_image=True, detect_method=detect.extractSURFFeatures, matches_display_num=100)

m000_P15=test.findMatches(imgP15,img000);
m000_N15=test.findMatches(imgN15,img000);
m000_P30=test.findMatches(imgP30,img000);
m000_N30=test.findMatches(imgN30,img000);
m000_P45=test.findMatches(imgP45,img000);
m000_N45=test.findMatches(imgN45,img000);
m000_180=test.findMatches(img180,img000);
m000_false=test.findMatches(img180,imgFalse);

#nbs=test.testFindNeighbors(imgP15)

#
#
print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","false"))
##test different threshold
for d in range(20, 101, 10):
    fm000_P15=test.filterFP(m000_P15,d);
    fm000_N15=test.filterFP(m000_N15,d);
    fm000_P30=test.filterFP(m000_P30,d);
    fm000_N30=test.filterFP(m000_N30,d);
    fm000_P45=test.filterFP(m000_P45,d);
    fm000_N45=test.filterFP(m000_N45,d);
    fm000_180=test.filterFP(m000_180,d);
    fm000_false=test.filterFP(m000_false,d);
    print("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % (d, len(fm000_P15), len(fm000_N15), len(fm000_P30), len(fm000_N30),len(fm000_P45), len(fm000_N45), len(fm000_180),len(fm000_false)))
#
#print("\nRatio test (dist_threshold=50)")
#print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("ratio", "+15", "-15", "+30","-30","+45","-45","180"))
#
#for rt in np.arange(0.5, 1, 0.1):
#    rm000_P15=test.testMatchWithDistanceAndRatio(imgP15, img000, detect_method=detect.extractORBFeatures, distance_threshold=50, ratio_threshold=rt)
#    rm000_N15=test.testMatchWithDistanceAndRatio(imgN15, img000, detect_method=detect.extractORBFeatures, distance_threshold=50, ratio_threshold=rt)
#    rm000_P30=test.testMatchWithDistanceAndRatio(imgP30, img000, detect_method=detect.extractORBFeatures, distance_threshold=50, ratio_threshold=rt)
#    rm000_N30=test.testMatchWithDistanceAndRatio(imgN30, img000, detect_method=detect.extractORBFeatures, distance_threshold=50, ratio_threshold=rt)
#    rm000_P45=test.testMatchWithDistanceAndRatio(imgP45, img000, detect_method=detect.extractORBFeatures, distance_threshold=50, ratio_threshold=rt)
#    rm000_N45=test.testMatchWithDistanceAndRatio(imgN45, img000, detect_method=detect.extractORBFeatures, distance_threshold=50, ratio_threshold=rt)
#    rm000_180=test.testMatchWithDistanceAndRatio(img180, img000, detect_method=detect.extractORBFeatures, distance_threshold=50, ratio_threshold=rt)
#    print("%.1f\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % (rt, len(rm000_P15), len(rm000_N15), len(rm000_P30), len(rm000_N30),len(rm000_P45), len(rm000_N45), len(rm000_180)))


#print("\nDistance test (ratio_threshold=1.1)")
#print("%s\t%s\t%s\t%s\t%s\t%s" % ("dist", "+15", "-15", "+30","-30","180"))
#si = False
##test different threshold
#for d in range(20, 101, 10):
#    fm000_P15=test.testMatchWithDistanceAndRatio(imgP15, img000, distance_threshold=d, ratio_threshold=1.1, show_image=si, matches_display_num=100);
#    fm000_N15=test.testMatchWithDistanceAndRatio(imgN15, img000, distance_threshold=d, ratio_threshold=1.1, show_image=si, matches_display_num=100);
#    fm000_P30=test.testMatchWithDistanceAndRatio(imgP30, img000, distance_threshold=d, ratio_threshold=1.1, show_image=si, matches_display_num=100);
#    fm000_N30=test.testMatchWithDistanceAndRatio(imgN30, img000, distance_threshold=d, ratio_threshold=1.1, show_image=si, matches_display_num=100);
#    fm000_180=test.testMatchWithDistanceAndRatio(img180, img000, distance_threshold=d, ratio_threshold=1.1, show_image=si, matches_display_num=100);
#    print("%d\t%d\t%d\t%d\t%d\t%d" % (d, len(fm000_P15), len(fm000_N15), len(fm000_P30), len(fm000_N30), len(fm000_180)))
#test.testWeightedMatching(img1, img2, 60, 0, d, detect_method=detect.extractSURFFeatures)

"""
#KeyPoint Parameters
#_pt	x & y coordinates of the keypoint
#_size	keypoint diameter
#_angle	keypoint orientation
#_response	keypoint detector response on the keypoint (that is, strength of the keypoint)
#_octave	pyramid octave in which the keypoint has been detected
#_class_id	object id
"""
#for k in kp1:
#    print(k.octave)

si = False  #show_image
bt = 0.5    #blocked_threshold
nn = 5     #neighbor_number
print("\nAdjusted confidence test distance(ratio=%.2f, neighbor_number=%d)"%(bt,nn))
print("%-5s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","False"))
#test different threshold
for d in range(40, 101, 10):
    con000_P15=test.testAdjustedConfidence(imgP15, img000, distance_threshold=d, neighbor_num=nn, h_angle=15, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_N15=test.testAdjustedConfidence(imgN15, img000, distance_threshold=d, neighbor_num=nn, h_angle=-15, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_P30=test.testAdjustedConfidence(imgP30, img000, distance_threshold=d, neighbor_num=nn, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_N30=test.testAdjustedConfidence(imgN30, img000, distance_threshold=d, neighbor_num=nn, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_P45=test.testAdjustedConfidence(imgP45, img000, distance_threshold=d, neighbor_num=nn, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_N45=test.testAdjustedConfidence(imgN45, img000, distance_threshold=d, neighbor_num=nn, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_180=test.testAdjustedConfidence(img180, img000, distance_threshold=d, neighbor_num=nn, h_angle=180, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_false=test.testAdjustedConfidence(imgFalse, img000, distance_threshold=d, neighbor_num=nn, h_angle=0, show_image=si, matches_display_num=100, blocked_threshold=bt);
    print("%d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d" % (d, con000_P15[0],con000_P15[2],con000_P15[1], con000_N15[0],con000_N15[2],con000_N15[1], con000_P30[0],con000_P30[2],con000_P30[1], con000_N30[0],con000_N30[2],con000_N30[1],con000_P45[0],con000_P45[2],con000_P45[1], con000_N45[0],con000_N45[2],con000_N45[1], con000_180[0],con000_180[2],con000_180[1],con000_false[0],con000_false[2],con000_false[1]))

exit()



si = False
dis = 100
nn = 5     #neighbor_number
print("\nAdjusted confidence test ratio(distance=%d, neighbor_number=%d)"%(dis,nn))
print("%-5s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","False"))
#test different threshold
for bt in np.arange(0.2, 1.1, 0.1):
    con000_P15=test.testAdjustedConfidence(imgP15, img000, distance_threshold=dis, neighbor_num=nn, h_angle=15, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_N15=test.testAdjustedConfidence(imgN15, img000, distance_threshold=dis, neighbor_num=nn, h_angle=-15, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_P30=test.testAdjustedConfidence(imgP30, img000, distance_threshold=dis, neighbor_num=nn, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_N30=test.testAdjustedConfidence(imgN30, img000, distance_threshold=dis, neighbor_num=nn, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_P45=test.testAdjustedConfidence(imgP45, img000, distance_threshold=dis, neighbor_num=nn, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_N45=test.testAdjustedConfidence(imgN45, img000, distance_threshold=dis, neighbor_num=nn, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_180=test.testAdjustedConfidence(img180, img000, distance_threshold=dis, neighbor_num=nn, h_angle=180, show_image=si, matches_display_num=100, blocked_threshold=bt);
    con000_false=test.testAdjustedConfidence(imgFalse, img000, distance_threshold=dis, h_angle=0, show_image=si, matches_display_num=100, blocked_threshold=bt);
#print("%.2f\t%d\t%d\t%d\t%d\t%d\t%d" % (bt, con000_P15[1], con000_N15[1], con000_P30[1], con000_N30[1], con000_180[1],con000_false[1]))
    print("%.02f\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d" % (bt, con000_P15[0],con000_P15[2],con000_P15[1], con000_N15[0],con000_N15[2],con000_N15[1], con000_P30[0],con000_P30[2],con000_P30[1], con000_N30[0],con000_N30[2],con000_N30[1],con000_P45[0],con000_P45[2],con000_P45[1], con000_N45[0],con000_N45[2],con000_N45[1], con000_180[0],con000_180[2],con000_180[1],con000_false[0],con000_false[2],con000_false[1]))

#si = True
dis = 100
bt = 0.5    #blocked_threshold
print("\nAdjusted confidence test neighbor number(ratio=%.2f, distance=%d)"%(bt,dis))
print("%-5s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s" % ("dist", "+15", "-15", "+30","-30","+45","-45","180","False"))
#test different threshold
for nn in np.arange(4, 11, 1):
    con000_P15=test.testAdjustedConfidence(imgP15, img000, distance_threshold=dis, h_angle=15, show_image=si, matches_display_num=100, blocked_threshold=bt, neighbor_num=nn);
    con000_N15=test.testAdjustedConfidence(imgN15, img000, distance_threshold=dis, h_angle=-15, show_image=si, matches_display_num=100, blocked_threshold=bt, neighbor_num=nn);
    con000_P30=test.testAdjustedConfidence(imgP30, img000, distance_threshold=dis, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=bt, neighbor_num=nn);
    con000_N30=test.testAdjustedConfidence(imgN30, img000, distance_threshold=dis, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=bt, neighbor_num=nn);
    con000_P45=test.testAdjustedConfidence(imgP45, img000, distance_threshold=dis, h_angle=30, show_image=si, matches_display_num=100, blocked_threshold=bt, neighbor_num=nn);
    con000_N45=test.testAdjustedConfidence(imgN45, img000, distance_threshold=dis, h_angle=-30, show_image=si, matches_display_num=100, blocked_threshold=bt, neighbor_num=nn);
    con000_180=test.testAdjustedConfidence(img180, img000, distance_threshold=dis, h_angle=180, show_image=si, matches_display_num=100, blocked_threshold=bt, neighbor_num=nn);
    con000_false=test.testAdjustedConfidence(imgFalse, img000, distance_threshold=dis, h_angle=0, show_image=si, matches_display_num=100, blocked_threshold=bt, neighbor_num=nn);
#print("%d\t%d\t%d\t%d\t%d\t%d\t%d" % (nn, con000_P15[1], con000_N15[1], con000_P30[1], con000_N30[1], con000_180[1],con000_false[1]))
    print("%d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d\t%4.02f:%.02f:%-6d" % (nn, con000_P15[0],con000_P15[2],con000_P15[1], con000_N15[0],con000_N15[2],con000_N15[1], con000_P30[0],con000_P30[2],con000_P30[1], con000_N30[0],con000_N30[2],con000_N30[1],con000_P45[0],con000_P45[2],con000_P45[1], con000_N45[0],con000_N45[2],con000_N45[1], con000_180[0],con000_180[2],con000_180[1],con000_false[0],con000_false[2],con000_false[1]))
"""

"""
