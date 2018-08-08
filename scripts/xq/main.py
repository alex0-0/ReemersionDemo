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

#img1 = cv2.imread("000.JPG")
#img2 = cv2.imread("090.JPG")
img1 = cv2.imread("-30.JPG")
img2 = cv2.imread("000.JPG")


img000 = cv2.imread("000.JPG")
imgP15 = cv2.imread("015.JPG")
imgN15 = cv2.imread("-15.JPG")
imgP30 = cv2.imread("030.JPG")
imgN30 = cv2.imread("-30.JPG")
img180 = cv2.imread("180.JPG")

print(img1.shape)
print(img2.shape)

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
m000_180=test.findMatches(img180,img000);


print("%s\t%s\t%s\t%s\t%s\t%s" % ("dist", "+15", "-15", "+30","-30","180"))
#test different threshold
for d in range(20, 101, 10):
    fm000_P15=test.filterFP(m000_P15,d);
    fm000_N15=test.filterFP(m000_N15,d);
    fm000_P30=test.filterFP(m000_P30,d);
    fm000_N30=test.filterFP(m000_N30,d);
    fm000_180=test.filterFP(m000_180,d);
    print("%d\t%d\t%d\t%d\t%d\t%d" % (d, len(fm000_P15), len(fm000_N15), len(fm000_P30), len(fm000_N30), len(fm000_180)))


#test.testWeightedMatching(img1, img2, 60, 0, d, detect_method=detect.extractSURFFeatures)

"""
KeyPoint Parameters 
_pt	x & y coordinates of the keypoint
_size	keypoint diameter
_angle	keypoint orientation
_response	keypoint detector response on the keypoint (that is, strength of the keypoint)
_octave	pyramid octave in which the keypoint has been detected
_class_id	object id
"""
#for k in kp1:
#    print(k.octave)

test.testDetect(img1, detect_method=detect.extractSURFFeatures)
test.testDetect(img2, detect_method=detect.extractSURFFeatures)
