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
    cv2.waitKey(0)

    #display detected feature points
    t = "image with key points"
    if title != None:
        t = title
    if DEBUG > 0:
        print(TAG + "key points number: " + str(len(kps)))
    cv2.namedWindow(t, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(t,600,600)
    cv2.imshow(t, img_kp)
    cv2.waitKey(0)

'''
    Assume 000.JPG is the template image in every directory.
    The file name is the change angle of that image
    The false image file name start with 'f' and is followed by the change angle, like f30.JPG
'''
def testAlgorithmPrecision():
    ds = ["lamp", "pops", "GrandfatherClock", "Motorcycle", "Robot", "horse", "DishSoap"]
    threshold = 0.3
    template_file = "000.JPG"

    answer = []
    ORB = []
    SURF = []
    FARES = []    #our algorithm
    for d in ds:
        t = cv2.imread(d + "/" + template_file)
        for f in os.listdir(d):
            if f == template_file:
                continue
            #read image
            img = cv2.imread(d + "/" + f)

            if f == "false.jpg":
                angle = 0
            else:
                #strip f mark in false image name
                if f.startswith('f'):
                    a = False
                    f = f[1:]
                else:
                    a = True
                try:   #let excpetion be discovered
                    angle = int(os.path.splitext(f)[0])
                except ValueError:
                    continue
            #if the image's angle change is more than 90 degree, we think it as a false image
            if abs(angle) >= 90:
                a = False
            answer.append(a)
            ORB.append(testMatchPrecision(img, t, detect.extractORBFeatures))
            SURF.append(testMatchPrecision(img, t, detect.extractSURFFeatures))
            FARES.append(testAdjustedConfidence(img, t, h_angle=angle))
#    answer = [True, False, True, False, True, True, False, True, True, False, True, False, False, True, False, False, True, True, False, False, False, True, True, True, True, False, True, False, True, False, False, True, False, True, True, True, True, True, True, False, False, False, True, False, True, True, True, False, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, False, True, True, False, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, False, True, False, True, True, False, True, True, False, True, True, True, False, True, True, True, True, True, False, True, True, True, True, False, False, False, True, True, False, False, False, True, False, True, False, True, True, False, True, True, True, True, False, True, False, False, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True]
#    FARES = [(0.8021390374331551, 126, 1), (0.36613923185549363, 79, 0.40231629987999246), (0.8287292817679558, 138, 1), (0.16826423128943208, 100, 0.3334323054567656), (1, 163, 0.6931302418439074), (1, 152, 0.7955851249577736), (0.18945605285958514, 70, 0.32539111475157506), (1, 126, 0.44983639732129527), (1, 135, 0.6591404555108603), (0, 112, 0.37618117801337786), (0.9664698499293102, 138, 0.7672414159526542), (0.2116645765962315, 82, 0.2891365784876626), (0, 156, 0.7061498441595411), (1, 156, 0.8489810628207684), (0.16543463485898047, 137, 0.25996871192125504), (0.11584824019887101, 61, 0.235450821515298), (1, 137, 1), (0.9662148508648218, 142, 0.92241311095895), (0.15276896509795998, 64, 0.29945464449516623), (0.3681517096251078, 144, 0.49364221704910877), (0.4326399034963043, 145, 0.3413048127581957), (0.8571428571428571, 150, 1), (0.872093023255814, 156, 1), (1, 176, 0.8156580030859978), (0.6649199895736924, 117, 0.3723163099513511), (0.46968128993145325, 149, 0.494042352687234), (0.44522025161588386, 159, 0.32029558185868434), (0.1490382048884361, 113, 0.15554958277191147), (1, 151, 1), (0.3287741205990582, 161, 0.43317907110479487), (0.11699126448397845, 181, 0.1492808534815565), (0.41023472888190976, 143, 0.39851373662814094), (0.8607569896215673, 144, 0.42737725007709615), (1, 152, 1), (0.7343342036553524, 117, 1), (0.8086253369272237, 129, 1), (0.7013715710723192, 99, 1), (0.6667654467328493, 103, 1), (0.819672131147541, 134, 1), (0.9857678967994299, 138, 0.5977353076070245), (0.2737434302441356, 145, 0.3191425869841318), (0.3461911867107249, 90, 0.6499010373232474), (1, 160, 1), (0.2404057246900658, 147, 0.375500977060147), (1, 133, 1), (1, 153, 1), (1, 146, 1), (0.1433645862053485, 160, 0.2111741356439639), (0.9305210918114143, 128, 1), (1, 136, 1), (0.6814337140670403, 124, 0.2511951730286345), (0.9054325955734407, 145, 1), (1, 146, 1), (0.9151812140676685, 106, 0.4427027603961466), (1, 163, 1), (1, 147, 1), (0.8640552995391706, 128, 1), (0.6947660954145438, 119, 1), (1, 145, 1), (0.803106936416185, 154, 1), (0.7785545954062788, 179, 0.8330534170847184), (0.7834261838440112, 141, 1), (0.7352941176470589, 140, 1), (0.7905976388584054, 138, 0.9578190939315354), (0.31025529986845024, 98, 0.48062670731066276), (0.7905782851222358, 160, 0.6409075272807825), (0.1182581686100205, 70, 0.3302013798851222), (0.9209021636727345, 153, 0.7069757760938913), (0.887957750214027, 138, 0.7631545716464335), (0.7053259445960538, 150, 0.555249843924019), (0.9596491852084443, 156, 0.6764740158026739), (0.7683676776294087, 160, 0.7547078078048858), (0.6432689042706871, 136, 0.7127850263456015), (0.6318591513566079, 127, 0.686123619959286), (0.610121493846075, 161, 0.6702862522670074), (0.6973538233544051, 148, 0.6276508610014543), (1, 147, 0.5842101269693559), (0.5567291025956826, 141, 0.6248808854771647), (0.6551401666537244, 173, 0.6077470482149443), (0.7377232038830351, 145, 0.8382451585045003), (0.7212694342041994, 133, 1), (0.8006595666463303, 141, 0.7318992196033237), (0.7769337016574586, 138, 1), (0.5271499073722619, 138, 0.4930962957849065), (0.8403361344537815, 143, 1), (0.8223684210526315, 158, 1), (0.8021390374331551, 170, 1), (0.821917808219178, 135, 1), (0.7922252010723861, 127, 1), (1, 158, 1), (0.5364208053782663, 140, 0.9152203314510705), (1, 137, 1), (1, 143, 1), (0.5754737379600844, 107, 0.9618763594611937), (1, 154, 1), (0.9861932938856016, 149, 1), (0.3646929168839463, 156, 0.40998157976495925), (1, 126, 1), (0.9509721048182586, 162, 1), (1, 169, 1), (1, 116, 0.8722238459006078), (1, 166, 1), (1, 162, 1), (0.8758271701050993, 133, 1), (0.7717372663351054, 157, 1), (1, 118, 1), (0.6328788629681862, 125, 0.14254028445229416), (0.6410498624707294, 140, 0.7692598349648753), (0.43613464842790667, 153, 0.5044624100149454), (0.4420299713723514, 156, 0.23357651329045911), (0.4180047347294064, 170, 0.18767559518463148), (0.022858499940183408, 119, 0.04466199219081989), (0.15747440380080827, 130, 0.2044404540571897), (0.1091394754725182, 150, 0.18364815584317967), (0.24237075253271648, 163, 0.1633578872070509), (0.29202581917535253, 126, 0.23437265315790093), (0.0733908359904235, 145, 0.11896688025844905), (0.05493800335666867, 141, 0.05843775764457497), (0.1089715547137194, 118, 0.1758131061430574), (0.18807983830474653, 159, 0.17668106022567098), (0.48022683985186576, 102, 0.22958592463788896), (0.3430537403604623, 153, 0.33914429602587015), (0.1336561441871339, 151, 0.1695337280629118), (0.7156488549618321, 107, 1), (0.6787330316742082, 110, 1), (0.5180980430331339, 143, 0.41102444747295286), (0.8333333333333334, 140, 1), (0.7520053475935828, 126, 1), (0.6684491978609626, 104, 1), (0.8264462809917356, 137, 1), (0.5070638978222104, 127, 0.620695657390432), (1, 139, 0.9393244941014425), (0.41286587607028025, 147, 0.3982012411278933), (0.23658995015066783, 136, 0.28706247284947695), (1, 161, 1), (0.06776079888380357, 157, 0.0929678160685785), (1, 137, 1), (0.11107963148382208, 134, 0.08330972361286657), (1, 132, 1), (1, 132, 1), (1, 147, 1), (1, 171, 1), (0.9380863039399625, 131, 1), (0.97911227154047, 117, 1), (0.9054325955734407, 145, 1), (1, 163, 1), (1, 169, 1), (1, 163, 1), (0.8370535714285714, 116, 1), (1, 133, 1)]
#    ORB = [0.018, 0.002, 0.028, 0.014, 0.01, 0.01, 0.002, 0.006, 0.006, 0.016, 0.002, 0.01, 0.018, 0.012, 0.014, 0.0, 0.016, 0.004, 0.01, 0.006, 0.0, 0.554, 0.464, 0.014, 0.008, 0.004, 0.014, 0.002, 0.082, 0.002, 0.0, 0.086, 0.0, 0.33, 0.106, 0.098, 0.188, 0.326, 0.056, 0.004, 0.0, 0.002, 0.034, 0.0, 0.034, 0.044, 0.026, 0.004, 0.038, 0.048, 0.0, 0.072, 0.054, 0.002, 0.054, 0.044, 0.042, 0.18, 0.042, 0.042, 0.022, 0.048, 0.166, 0.016, 0.004, 0.0, 0.004, 0.0, 0.0, 0.0, 0.0, 0.004, 0.0, 0.002, 0.012, 0.002, 0.0, 0.004, 0.006, 0.008, 0.154, 0.01, 0.264, 0.0, 0.224, 0.294, 0.432, 0.21, 0.002, 0.062, 0.008, 0.076, 0.062, 0.002, 0.06, 0.126, 0.002, 0.118, 0.134, 0.05, 0.0, 0.082, 0.08, 0.142, 0.438, 0.078, 0.006, 0.022, 0.024, 0.002, 0.004, 0.002, 0.0, 0.002, 0.002, 0.004, 0.006, 0.0, 0.004, 0.002, 0.0, 0.002, 0.002, 0.444, 0.558, 0.0, 0.264, 0.322, 0.5, 0.338, 0.004, 0.048, 0.006, 0.002, 0.046, 0.0, 0.096, 0.0, 0.068, 0.102, 0.05, 0.038, 0.228, 0.156, 0.164, 0.058, 0.118, 0.082, 0.268, 0.114]
#    SURF = [0.09958190801976435, 0.004561003420752566, 0.10262257696693272, 0.008361839604713038, 0.039908779931584946, 0.030026605853287723, 0.010642341315089319, 0.043329532497149374, 0.035727860129228434, 0.0395286963131889, 0.04294944887875333, 0.009502090459901177, 0.04104903078677309, 0.04104903078677309, 0.04142911440516914, 0.018244013683010263, 0.06423413150893197, 0.06841505131128849, 0.008741923223109084, 0.013135946622185155, 0.006046705587989992, 0.41492910758965806, 0.3035863219349458, 0.019599666388657216, 0.011259382819015847, 0.009799833194328608, 0.03544620517097581, 0.014595496246872394, 0.10758965804837364, 0.0025020850708924102, 0.011676396997497914, 0.12760633861551293, 0.005838198498748957, 0.274603836530442, 0.17950963222416813, 0.2180385288966725, 0.2635726795096322, 0.3380035026269702, 0.14448336252189142, 0.0297723292469352, 0.04115586690017513, 0.043782837127845885, 0.10507880910683012, 0.05779334500875657, 0.0893169877408056, 0.09807355516637478, 0.0989492119089317, 0.03152364273204904, 0.1085814360770578, 0.10332749562171628, 0.09982486865148861, 0.18476357267950963, 0.1243432574430823, 0.047285464098073555, 0.1637478108581436, 0.13047285464098074, 0.11996497373029773, 0.2478108581436077, 0.10595446584938704, 0.08867667121418826, 0.06275579809004093, 0.10140973169622555, 0.19417917235106866, 0.05593451568894952, 0.008185538881309686, 0.021828103683492497, 0.0068212824010914054, 0.017735334242837655, 0.023192360163710776, 0.018190086402910415, 0.023192360163710776, 0.02819463392451114, 0.007276034561164165, 0.02773988176443838, 0.04456571168713051, 0.030468394724874944, 0.008185538881309686, 0.03638017280582083, 0.02364711232378354, 0.03865393360618463, 0.1914506593906321, 0.0241018644838563, 0.24439918533604887, 0.03665987780040733, 0.24439918533604887, 0.269857433808554, 0.3564154786150713, 0.23116089613034624, 0.04786150712830957, 0.15071283095723015, 0.04276985743380855, 0.1334012219959267, 0.12627291242362526, 0.04073319755600815, 0.14969450101832993, 0.164969450101833, 0.05091649694501019, 0.1395112016293279, 0.20977596741344195, 0.15580448065173116, 0.03054989816700611, 0.18737270875763748, 0.1710794297352342, 0.18329938900203666, 0.33706720977596744, 0.15376782077393075, 0.029796839729119638, 0.11693002257336343, 0.11557562076749435, 0.014898419864559819, 0.017155756207674944, 0.01038374717832957, 0.03250564334085779, 0.025733634311512415, 0.02167042889390519, 0.016704288939051917, 0.015349887133182845, 0.01038374717832957, 0.017607223476297968, 0.032054176072234764, 0.017155756207674944, 0.03702031602708804, 0.03160270880361174, 0.33200953137410644, 0.38204924543288327, 0.046068308181096106, 0.2764098490865767, 0.33439237490071483, 0.4344718030182685, 0.2533756949960286, 0.050833995234312944, 0.06671961874503574, 0.05242255758538523, 0.057188244638602066, 0.11517077045274027, 0.019857029388403495, 0.0976965845909452, 0.03891977760127085, 0.08260524225575853, 0.10246227164416204, 0.07625099285146943, 0.10802223987291501, 0.17315329626687848, 0.13264495631453535, 0.2255758538522637, 0.13026211278792693, 0.19618745035742652, 0.15091342335186655, 0.2057188244638602, 0.11358220810166798]
    #validate
    predictions = [
                    [True if p[0] > threshold else False for p in FARES],
                    [True if p > threshold else False for p in ORB],
                    [True if p > threshold else False for p in SURF]
                    ]
#    print(answer)
#    print(FARES)
#    print(ORB)
#    print(SURF)

    num = len(answer)
    positive_num = len([1 for x in answer if x])
    print("FARES average true positive score: " + str(sum([a[0] for [a, b] in zip(FARES, answer) if b])/positive_num))
    print("ORB average true positive score: " + str(sum([a for [a, b] in zip(ORB, answer) if b])/positive_num))
    print("SURF average true positive score: " + str(sum([a for [a, b] in zip(SURF, answer) if b])/positive_num))
    print("FARES average detected blocked points: " + str(sum([a[2] for a in FARES])/len(FARES)))
    print("FARES average true negative score: " + str(sum([a[0] for [a, b] in zip(FARES, answer) if not b])/(num -positive_num)))
    print("ORB average true negative score: " + str(sum([a for [a, b] in zip(ORB, answer) if not b])/(num -positive_num)))
    print("SURF average true negative score: " + str(sum([a for [a, b] in zip(SURF, answer) if not b])/(num -positive_num)))
    for p in predictions:
        #true positive
        tp = [1 for a,b in zip(p, answer) if a and b]
        precision = len(tp)/len([1 for a in p if a])
        recall = len(tp)/positive_num
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        '''
        160 files
FARES average true positive score: 0.5623564880875342
ORB average true positive score: 0.07134666666666667
SURF average true positive score: 0.09127473018172347
FARES average detected blocked points: 0.7121097479666648
precision: 0.8133333333333334
recall: 0.6533333333333333
precision: 0.4
recall: 0.07333333333333333
precision: 0.38666666666666666
recall: 0.06
        '''
#    print(ORB)
#    print(SURF)
#    print(pre)
#    print(answer)
 


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

    #matches = sorted(matches, key=lambda m:m.distance)
    #matches = matches[:int(0.5*len(kp2))]
    
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
def testAdjustedConfidence(query_img, template_img, h_angle=0, v_angle=0, distance_threshold=0.5, blocked_threshold=0.5, neighbor_num=5, detect_method=detect.extractORBFeatures, show_image=False, matches_display_num=0, pos_dis=150):
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
#    filtered_matches = [m for m in matches if m.distance<100]
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
