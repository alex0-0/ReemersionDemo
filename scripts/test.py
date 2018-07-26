#!/usr/bin/env python3
""" Test functions to work """
import cv2
import detect
import numpy as np
import distort
import match
from match import DescriptorType
import os
import matplotlib.pyplot as plt
import glob

DEBUG = True
TAG = "TEST\t"

def trackDistort(img, des, distort_method, detect_method=detect.extractORBFeatures):
    t = distort_method(img)
    features = []
    matches = []
    if detect_method == detect.extractSURFFeatures:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    for i in t:
        k, d = detect_method(i)
        if len(k) == 0:
            features.append(0)
            matches.append(0)
            continue
        m = bf.match(des, d)
        features.append(len(k))
        matches.append(len(m))

    return features, matches

#track the change on the number of feature points. des is descriptor of original image, used for matching, if not given, will calculate in method
def trackRotate(img, detect_method=detect.extractORBFeatures, des=None):
    if des is None:
        kp, des = detect_method(img)
    return trackDistort(img, des, distort.rotateImage, detect_method)

def trackScale(img, detect_method=detect.extractORBFeatures, des=None):
    if des is None:
        kp, des = detect_method(img)
    return trackDistort(img, des, distort.scaleImage, detect_method)

def trackAffine(img, detect_method=detect.extractORBFeatures, des=None):
    if des is None:
        kp, des = detect_method(img)
    return trackDistort(img, des, distort.affineImage, detect_method)

def trackPerspective(img, detect_method=detect.extractORBFeatures, des=None):
    if des is None:
        kp, des = detect_method(img)
        print("the number of feature points: " + str(len(kp)))
    return trackDistort(img, des, distort.changeImagePerspective, detect_method)

def trackFeatureChange(img, angle_step, scale_step, affine_step, pers_step, detect_method=detect.extractORBFeatures):
    kp1, des1 = detect_method(img)

    print(TAG + "original: " + str(len(kp1)))

    print(TAG + "**************rotate**************")
    k, m = trackRotate(img, detect_method, des1)
    for i in range(len(k)):
        print(TAG + ("+" if (i%2==0) else "-") + str((i//2+1)*angle_step) + ": " + str(len(k)))
        print(TAG + "matched features: " + str(len(m)))

    print(TAG + "**************scale**************")
    k, m = trackScale(img, detect_method, des1)
    for i in range(len(k)):
        scale = (1 + (i//2+1)*scale_step) if (i%2==0) else (1 - (i//2+1)*scale_step)
        print(TAG + str(scale) + ": " + str(len(k)))
        print(TAG + "matched features: " + str(len(m)))

    print(TAG + "**********scale + rotate**********")
    t = distort.scaleImage(img)
    if detect_method == detect.extractSURFFeatures:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    count = 0
    for i in t:
        scale = (1 + (count//2+1)*scale_step) if (count%2==0) else (1 - (count//2+1)*scale_step)
        print(TAG + str(scale) + ": ")
        im = distort.rotateImage(i)
        c = 0
        for k in im:
            kp, des = detect_method(k)
            print(TAG + ("+" if (c%2==0) else "-") + str((c//2+1)*angle_step) + ": " + str(len(kp)))
            m = bf.match(des1, des)
            print(TAG + "matched features: " + str(len(m)))
            c += 1
        count += 1

    print(TAG + "**************affine**************")
    k, m = trackAffine(img, detect_method, des1)
    for i in range(len(k)):
        print(TAG + str(i) + ": " + str(len(k)))
        print(TAG + "matched features: " + str(len(m)))

    print(TAG + "perspective:")
    k, m = trackPerspective(img, detect_method, des1)
    for i in range(len(k)):
        print(TAG + str(i) + ": " + str(len(kp)))
        print(TAG + "matched features: " + str(len(m)))

def testDetect(img, detect_method=detect.extractORBFeatures, title=None):
    kps, des = detect_method(img)
    img_kp = cv2.drawKeypoints(img, kps, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #display detected feature points
    t = "image with key points"
    if title != None:
        t = title
    cv2.imshow(t, img_kp)
    cv2.waitKey(0)

def testDistinctFeature(img, detect_method=detect.extractORBFeatures):
    kps, des = detect.extractDistinctFeatures(img, detect_method)
    if DEBUG:
        print(TAG + "distinct feature points: " + str(len(kps)))
    img_kp = cv2.drawKeypoints(img, kps, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("image with distinct key points", img_kp)
    cv2.waitKey(0)

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
    if DEBUG:
        print(TAG + "matches size: " + str(len(matches)))
        print(TAG + "query key points size: " + str(len(kp1)))
        print(TAG + "train key points size: " + str(len(kp2)))
    
    match.drawMatches(img_1,kp1,img_2,kp2,matches[:10], thickness=3, color=(255,0,0))

"Track feature points changes of all pictures in directory d"
def massTrackFeaturePoints(d, angle_step, scale_step, detect_method=detect.extractORBFeatures):
    if os.path.isdir(d) == False:
        print(TAG + d + " is not a directory")
        return 
    #declare those variables for recording features change
    original = 0
    feature_rotate = None
    match_rotate = None
    feature_scale = None
    match_scale = None
    feature_affine = None
    match_affine = None
    feature_per = None
    match_per = None

    images = [os.path.join(d,name) for name in os.listdir(d) if os.path.isfile(os.path.join(d, name))]

    #the number of files in directory d
    file_num = len(images)

    for f in  images:
        if DEBUG:
            print(TAG + "file name: " + str(f))
        image = cv2.imread(f)
        kp, des = detect_method(image)
        original += len(kp)

        f1, m1 = trackRotate(image, detect_method)
        #initiate record array
        if feature_rotate is None:
            feature_rotate = np.array([0]*len(f1))
            match_rotate = np.array([0]*len(f1))
        feature_rotate += np.array(f1)
        match_rotate += np.array(m1)

        f2, m2 = trackScale(image, detect_method)
        #initiate record array
        if feature_scale is None:
            feature_scale = np.array([0]*len(f2))
            match_scale = np.array([0]*len(f2))
        feature_scale += np.array(f2)
        match_scale += np.array(m2)

        f3, m3 = trackAffine(image, detect_method)
        #initiate record array
        if feature_affine is None:
            feature_affine = np.array([0]*len(f3))
            match_affine = np.array([0]*len(f3))
        feature_affine += np.array(f3)
        match_affine += np.array(m3)

        f4, m4 = trackPerspective(image, detect_method)
        #initiate record array
        if feature_per is None:
            feature_per = np.array([0]*len(f4))
            match_per = np.array([0]*len(f4))
        feature_per += np.array(f4)
        match_per += np.array(m4)

    #calculate average value
    original = original / file_num
    feature_rotate = feature_rotate / file_num
    match_rotate = match_rotate / file_num
    feature_scale = feature_scale / file_num
    match_scale = match_scale / file_num
    feature_affine = feature_affine / file_num
    match_affine = match_affine / file_num
    feature_per = feature_per / file_num
    match_per = match_per / file_num

    if DEBUG:
        print(TAG + "length of feature_rotate: " + str(len(feature_rotate)))
        print(TAG + "feature_rotate: " + str(feature_rotate))
        print(TAG + "length of match_rotate: " + str(len(match_rotate)))
        print(TAG + "match_rotate: " + str(match_rotate))
        print(TAG + "length of feature_scale: " + str(len(feature_scale)))
        print(TAG + "feature_scale: " + str(feature_scale))
        print(TAG + "length of match_scale: " + str(len(match_scale)))
        print(TAG + "match_scale: " + str(match_scale))
        print(TAG + "length of feature_affine: " + str(len(feature_affine)))
        print(TAG + "feature_affine: " + str(feature_affine))
        print(TAG + "length of match_affine: " + str(len(match_affine)))
        print(TAG + "match_affine: " + str(match_affine))
        print(TAG + "length of feature_per: " + str(len(feature_per)))
        print(TAG + "feature_per: " + str(feature_per))
        print(TAG + "length of match_per: " + str(len(match_per)))
        print(TAG + "match_per: " + str(match_per))

    #rearrange rotate and scale list
    fr = [0] * (len(feature_rotate) + 1)
    mr = [0] * (len(match_rotate) + 1)
    fr[int(len(feature_rotate)/2)] = original
    mr[int(len(match_rotate)/2)] = original
    pivot = int(len(feature_rotate)/2)
    for i in range(int(len(feature_rotate)/2)):
        fr[pivot-(i+1)] = feature_rotate[2*i]
        fr[pivot+(i+1)] = feature_rotate[2*i+1]
        mr[pivot-(i+1)] = match_rotate[2*i]
        mr[pivot+(i+1)] = match_rotate[2*i+1]
    feature_rotate = fr
    match_rotate = mr

    fs = [0] * (len(feature_scale) + 1)
    ms = [0] * (len(match_scale) + 1)
    fs[int(len(feature_scale)/2)] = original
    ms[int(len(match_scale)/2)] = original
    pivot = int(len(feature_scale)/2)
    for i in range(int(len(feature_scale)/2)):
        fs[pivot-(i+1)] = feature_scale[2*i]
        fs[pivot+(i+1)] = feature_scale[2*i+1]
        ms[pivot-(i+1)] = match_scale[2*i]
        ms[pivot+(i+1)] = match_scale[2*i+1]
    feature_scale = fs
    match_scale = ms

    feature_affine = np.insert(feature_affine, 0, original)
    match_affine = np.insert(match_affine, 0, original)
    feature_per = np.insert(feature_per, 0, original)
    match_per = np.insert(match_per, 0, original)
    if DEBUG:
        print(TAG + "length of feature_rotate: " + str(len(feature_rotate)))
        print(TAG + "feature_rotate: " + str(feature_rotate))

    x = np.array(list(range(-int(len(feature_rotate)/2), int(len(feature_rotate)/2)+1)))*angle_step
    plt.plot(x, feature_rotate, label = "feature points")
    plt.plot(x, match_rotate, label = "matches")
    plt.xlabel('rotated angle')
    plt.ylabel('the number of feature points or matches')
    plt.title('rotate')
    plt.legend()
    plt.show()

    x = np.array(list(range(-int(len(feature_scale)/2), int(len(feature_scale)/2)+1)))*scale_step
    plt.plot(x, feature_scale, label = "feature points")
    plt.plot(x, match_scale, label = "matches")
    plt.xlabel('scale')
    plt.ylabel('the number of feature points or matches')
    plt.title('scale')
    plt.legend()
    plt.show()

    x = list(range(len(feature_affine)))
    plt.plot(x, feature_affine, label = "feature points")
    plt.plot(x, match_affine, label = "matches")
    plt.xlabel('affine')
    plt.ylabel('the number of feature points or matches')
    plt.title('affine')
    plt.legend()
    plt.show()

    x = list(range(len(feature_per)))
    plt.plot(x, feature_per, label = "feature points")
    plt.plot(x, match_per, label = "matches")
    plt.xlabel('perspective')
    plt.ylabel('the number of feature points or matches')
    plt.title('perspective')
    plt.legend()
    plt.show()

def compareImageInSameCategory(img_name, d, detect_method=detect.extractORBFeatures, template_distinct_feature=False):
    if os.path.isdir(d) == False:
        print(TAG + d + " is not a directory")
        return 

    d_type = DescriptorType.ORB
    if detect_method == detect.extractSURFFeatures:
        d_type = DescriptorType.SURF

    img = cv2.imread(img_name)
    #extract distinct method
    if template_distinct_feature:
        kp, des = detect.extractDistinctFeatures(img, detect_method)
        #transfer to numpy array for matching
        des = np.array(des)
    else:
        kp, des = detect_method(img)
    if DEBUG:
        print(TAG + "length of template image's features: " + str(len(des)))
    #read files into list
    images = [cv2.imread(os.path.join(d,name)) for name in os.listdir(d) if os.path.isfile(os.path.join(d, name))]
    #get keypoints and descriptors of every image
    if DEBUG:
        [print(TAG + "ALERT! NULL IMAGE") for i in images if i is None]
    features = [detect_method(i) for i in images if i is not None]

    #get  matches
    matches = [match.matchFeature(des, kp, d, k, d_type) for (k,d) in features if d is not None]
#    matches = [match.BFMatchFeature(des, d, d_type) for (k,d) in features if d is not None]
    if DEBUG:
        [print(TAG + "number of matches: " + str(len(m))) for m in matches]
    print(TAG + "the number of feature points in template image: " + str(len(kp)))
    print(TAG + "average matched feature point for " + str(len(features)) + " images is: " + str(sum(len(m) for m in matches) / len(features)))
    print(TAG + "best matched image matches: " + str(len(max(matches,key=len))))
    print(TAG + "worst matched image matches:: " + str(len(min(matches,key=len))))
    if DEBUG:
        print(TAG + "display first image matches")
        match.drawMatches(img, kp, images[0], features[0][0], matches[0][:30], thickness=3, color=(255,0,0))


def checkDistinctFeatureInSameCategory(img_name, d, detect_method=detect.extractORBFeatures):
    compareImageInSameCategory(img_name, d, detect_method, True)
