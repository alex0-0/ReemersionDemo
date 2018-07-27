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

img1 = cv2.imread("t1.jpg")
img2 = cv2.imread("t2.jpg")

kp1, des1 = detect.extractORBFeatures(img1)
kp2, des2 = detect.extractORBFeatures(img2)
matches = match.BFMatchFeature(des1, des2, DescriptorType.ORB)
match.drawMatches(img1,kp1,img2,kp2,matches[:10], thickness=3, color=(255,0,0))
