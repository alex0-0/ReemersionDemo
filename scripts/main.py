#!/usr/bin/env python3
""" Call functions to work, program entry """
import cv2
import detect
import numpy as np
import distort
import match
import test

DEBUG = True
TAG = "MAIN\t"

img = cv2.imread("test.png")
#img1 = cv2.imread("distorted.png")
#
##save distorted image to default directory, i.e., distorted_img
#distort.saveDistortedImages(img)
#
#test.testMatch(img, img1, detect.extractORBFeatures)
#
##cv2.destroyAllWindows()
#
#test.trackFeatureChange(img, 5, 0.1, 0.1, 0.1, detect_method=detect.extractORBFeatures)
#
#if DEBUG:
#    img2 = cv2.imread("distorted_img/rotate_5.png")
#    test.testDetect(img, detect.extractORBFeatures)
#    test.testDetect(img2, detect.extractSURFFeatures)
#
#
##debug distinct feature points
#test.testDistinctFeature(img, detect.extractSURFFeatures)

#test.massTrackFeaturePoints("tmp/", 5, 0.1, detect.extractORBFeatures)
#img = cv2.imread("bike_template.JPEG")
#img1 = cv2.imread("bicycle/n02834778_9885.JPEG")
#test.testMatch(img, img1, detect.extractORBFeatures)

#test.compareImageInSameCategory("bike_template.JPEG", "bicycle", detect.extractORBFeatures)
test.checkDistinctFeatureInSameCategory("bike_template.JPEG", "bicycle", detect.extractORBFeatures)
