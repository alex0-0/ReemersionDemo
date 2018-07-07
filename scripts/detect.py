#!/usr/bin/env python3
""" Detect feature points """
import cv2
import distort


#return keypoints and descriptors
def extractFeatures(img):
    surf = cv2.xfeatures2d.SURF_create()
    #TODO: need to figure out how to increase the threshold
#    surf.hessianThreshold = 400
    keypoints, descriptors = surf.detectAndCompute(img, None)
    return keypoints, descriptors

def extractDistinctFeatures(img):
    #every feature points has to be matched to at least more than 3 pictures
    kDistinctThreshold = 3

    kps, des = extractFeatures(img)
    #count how many pictures the feature point exist
    counter = [0] * len(kps)

    distorted_img = []
    distorted_img.extend(distort.rotateImage(img))
    distorted_img.extend(distort.scaleImage(img))
    distorted_img.extend(distort.affineImage(img))
    distorted_img.extend(distort.changeImagePerspective(img))
    
    return distorted_img

#    distorted_features = [extractFeatures(i) for i in distorted_img]

    
