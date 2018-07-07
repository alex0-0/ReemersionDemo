#!/usr/bin/env python3
""" Distort images """
import numpy as np
import cv2 as cv

def rotateImage(img):
    #the step difference between angles of generating distorted images, in degree.
    kStepAngle = 5
    #the number of different scale distorted images
    kNum = 6
    rows,cols,ch = img.shape

    r = []
    for i in range(int(kNum/2)):
        M1 = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),kStepAngle*i,1)
        r.append(cv.warpAffine(img,M1,(cols,rows)))
        M2 = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),-kStepAngle*i,1)
        r.append(cv.warpAffine(img,M2,(cols,rows)))

    return r

def scaleImage(img):
    #the difference between scales of generating distorted images
    kStepScale = 0.1
    #the number of different rotated distorted images
    kNum = 6
    r = []

    for i in range(int(kNum/2)):
        #scale up
        r.append(cv.resize(img,None,fx=(1+i*kStepScale), fy=(1+i*kStepScale), interpolation = cv.INTER_CUBIC))
        #scale down
        r.append(cv.resize(img,None,fx=(1-i*kStepScale), fy=(1-i*kStepScale), interpolation = cv.INTER_CUBIC))

    return r

def affineImage(img):
    kStepAffine = 0.1
    kNum = 4
    rows,cols,ch = img.shape
    r = []
    
    pts1 = np.float32([[10,10],[200,50],[50,200]])
    for i in range(int(kNum/2)):
        pts2 = np.float32([[50+i*kStepAffine,100+i*kStepAffine],[200+i*kStepAffine,50+i*kStepAffine],[100+i*kStepAffine,250+i*kStepAffine]])
        #get matrix
        M1 = cv.getAffineTransform(pts1,pts2)
        M2 = cv.getAffineTransform(pts2,pts1)

        #affine image and add to list
        r.append(cv.warpAffine(img,M1,(cols,rows)))
        r.append(cv.warpAffine(img,M2,(cols,rows)))

    return r

def changeImagePerspective(img):
    kStepPer = 0.1
    kNum = 4
    rows,cols,ch = img.shape
    r = []
    
   # pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts1 = np.float32([[0,0],[cols,0],[cols,rows],[0,rows]])
    for i in range(int(kNum/2)):
        pts2 = np.float32([[0,i*kStepPer*rows],[cols,i*kStepPer*rows],[cols*(1-kStepPer*i),rows*(1-kStepPer*i)],[cols*i*kStepPer,rows*(1-kStepPer*i)]])
        #get matrix
        M1 = cv.getPerspectiveTransform(pts1,pts2)
        M2 = cv.getPerspectiveTransform(pts2,pts1)
        r.append(cv.warpPerspective(img,M1,(rows,cols)))
        r.append(cv.warpPerspective(img,M2,(rows,cols)))

    return r

def saveDistortedImages(img):
    t = rotateImage(img)
    c = 1
    for i in t:
        cv.imwrite("distorted_img/rotate_" + str(c) + ".png", i)
        c += 1

    t = scaleImage(img)
    c = 1
    for i in t:
        cv.imwrite("distorted_img/scale_" + str(c) + ".png", i)
        c += 1

    t = affineImage(img)
    c = 1
    for i in t:
        cv.imwrite("distorted_img/affine_" + str(c) + ".png", i)
        c += 1

    t = changeImagePerspective(img)
    c = 1
    for i in t:
        cv.imwrite("distorted_img/pers_" + str(c) + ".png", i)
        c += 1

    print("distorted images saved")
