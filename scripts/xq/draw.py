import matplotlib.pyplot as plt
import matplotlib.image as pim
import os
import test
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticks
import detect



TAG = 'DRAW\t'
DEGBUG = 0

def drawFigures(dire='horse'):
    def readImage(f):
        return plt.imread(dire+"/"+f)
    images = []
    images.append(readImage("000.JPG"))
    images.append(readImage("015.JPG"))
    images.append(readImage("-15.JPG"))
    images.append(readImage("030.JPG"))
    images.append(readImage("-30.JPG"))
    images.append(readImage("045.JPG"))
    images.append(readImage("-45.JPG"))
    images.append(readImage("180.JPG"))
    images.append(readImage("false.JPG"))
    f = plt.figure(figsize=(15,15))
    for i, img in enumerate(images):
        f.add_subplot(3, 3, i+1)
        plt.axis('off')
        plt.imshow(img)
    plt.subplots_adjust(left=0.125, right=0.6, top=0.99, bottom=0.51, hspace=0.00, wspace=0.00)
    plt.show()

def drawPrecisionChart(ds=['horse','lamp','pops']):
    fnames = []
    fnames.append("000.JPG")
    fnames.append("015.JPG")
    fnames.append("-15.JPG")
    fnames.append("030.JPG")
    fnames.append("-30.JPG")
    fnames.append("045.JPG")
    fnames.append("-45.JPG")
    fnames.append("180.JPG")
    fnames.append("false.JPG")

    keys = [15,-15,30,-30,45,-45,180,-9999999]

    def readImages(d):
        r = []
        for f in fnames:
            r.append(plt.imread(d+"/"+f))
        return r

    SURF = []
    ORB = []
    images = []
    for d in ds:
        images.append(readImages(d))
    for ig in images:
        SURF.append([test.testMatchPrecision(i, ig[0],detect_method=detect.extractSURFFeatures) for i in ig[1:]])
        ORB.append([test.testMatchPrecision(i, ig[0],detect_method=detect.extractORBFeatures) for i in ig[1:]])

    if DEGBUG > 1:
        print(TAG + "SURF precision list: " + str(SURF))
        print(TAG + "ORB precision list: " + str(ORB))
    
    ave_SURF = []
    ave_ORB = []
    for i, t in enumerate(SURF[0]):
        ave_SURF.append(sum([p[i] for p in SURF])/len(SURF))
        ave_ORB.append(sum([p[i] for p in ORB])/len(ORB))

    dic_SURF = dict(zip(keys, ave_SURF))
    dic_ORB = dict(zip(keys, ave_ORB))
#    dic = {15: 0.167, -15: 0.098, 30: 0.107, -30: 0.007, 45: 0.030, -45: 0.002, 180: 0.002, False: 0.002}
    k1 = sorted(dic_SURF)
#    k2 = sorted(dic_ORB)
    v1 = [dic_SURF[k] for k in k1]
    v2 = [dic_ORB[k] for k in k1]
    labels = [str(k) for k in k1]
    labels[0] = "False"

    #copied from tests
    #Adjusted confidence test number of matches(ratio=0.50, neighbor_number=5, pos_dis=150) ratio=1.0
    horse = np.array([0.75,0.59,0.71,0.39,0.51,0.30,0.08,0.10])
    lamp = np.array([0.91,0.96,1.00,0.96,0.75,0.72,0.10,0.03])
    pops = np.array([1.00,0.84,1.00,0.12,1.00,0.19,0.02,0.28])

    ave_score = (horse + lamp + pops)/len(ds)
    dic = dict(zip(keys, ave_score))
    ave_score = [dic[k] for k in k1]

    """
    In pyplot drawing function, the x stands for how x axis should be seperated, it it not necessary to use real key as x value
    To make the spacing equal, we can use list(range(len(keys))) to be the x value and then change the x labels.
    """
    fig, ax = plt.subplots(1,1)
    ax.plot(list(range(len(keys))), v1, label='SURF')
    ax.plot(list(range(len(keys))), v2, label='ORB')
    ax.plot(list(range(len(keys))), ave_score, label='our algorithm')
    ax.set_xticklabels([""]+labels)
    plt.xlabel("images")
    plt.ylabel("confidence")
    plt.legend(loc='best')

    plt.show()
##    labels = [item.get_text() for item in ax.get_xticklabels()]
##    [print(l) for l in labels]
##    plt.bar(list(range(len(labels))), values, tick_label=labels)
#
##    plt.plot(list(range(len(labels))), values)
#    
##    fig = plt.figure()
#    fig, ax = plt.subplots(1,1)
##    ax = fig.add_subplot(111)
##
#    ax.plot(list(range(len(keys))), values)
##    fig.canvas.draw()
###    labels = [item.get_text() for item in ax.get_xticklabels()]
##    [print(l) for l in labels]
##
###    labels = [item.get_text() for item in ax.get_xticklabels()]
###    plt.xticks([str(k) for k in keys],values)
###    ax.set_xticks(np.arange(len(values)))
##    ax.set_xticks(np.arange(len(keys)))
##
###    ax.xaxis.set_major_locator(mticks.LinearLocator(numticks=len(labels)))
##    ax.xaxis.set_major_locator(MaxNLocator(len(labels)))
#    ax.set_xticklabels([""]+labels)
#    plt.xlabel("images")
#    plt.ylabel("confidence")
#
#    plt.show()

def drawBlockedPointChart():
#Adjusted confidence test ratio(distance=0.50, neighbor_number=5, pos_dis=150)
    keys = [15,-15,30,-30,45,-45,180,False]
    x = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    lamp = [
            [180, 179, 159, 109, 88 , 77 , 62 , 34 , 20],
            [189, 187, 177, 136, 108, 66 , 37 , 23 , 10],
            [198, 196, 169, 127, 100, 68 , 44 , 21 , 13],
            [212, 201, 176, 119, 95 , 70 , 46 , 21 , 9 ],
            [212, 175, 130, 77 , 66 , 51 , 40 , 30 , 17],
            [221, 215, 175, 92 , 77 , 60 , 36 , 24 , 11],
            [230, 185, 127, 82 , 60 , 41 , 26 , 21 , 15],
            [172, 134, 92 , 70 , 53 , 42 , 37 , 17 , 16],
            ]
    horse = [
            [49, 49, 45, 35, 29, 19, 12, 5, 4],
            [53, 50, 49, 37, 33, 22, 20, 13, 11],
            [54, 49, 42, 39, 35, 29, 23, 14, 9],
            [53, 53, 49, 40, 37, 34, 26, 18, 10],
            [61, 51, 33, 27, 26, 23, 21, 15, 7],
            [53, 53, 52, 46, 28, 24, 15, 14, 12],
            [52, 23, 14, 14, 14, 14, 14, 14, 14],
            [53, 53, 51, 36, 23, 14, 9, 8, 5]
            ]
    pops = [
            [230, 219, 173, 138, 110, 91, 64, 25, 12],
            [239, 236, 197, 142, 130, 85, 58, 33, 14],
            [233, 213, 185, 140, 132, 86, 58, 32, 10],
            [293, 168, 109, 40, 34, 30, 29, 29, 18],
            [234, 215, 182, 146, 128, 100, 62, 39, 12],
            [297, 260, 149, 72, 54, 41, 25, 16, 13],
            [149, 60, 37, 35, 35, 17, 17, 17, 17],
            [269, 245, 201, 117, 94, 66, 47, 30, 12]
            ]

    fig, axs = plt.subplots(1,3, sharey=True)
    ds = [horse, lamp, pops]
    ts = ["horse", "lamp", "pops"]
    labels = [str(i) for i in x] 
    for i, d in enumerate(ds):
        for k, values in enumerate(ds[i]):
            axs[i].plot(x, values, label=str(keys[k]))
        axs[i].set_title(ts[i])
#    ax.plot(list(range(len(keys))), ave_score, label='score given by our algorithm')
    plt.legend(loc='best')
    fig.text(0.5, 0.04, 'ratio', ha='center', va='center')
    fig.text(0.06, 0.5, 'number of blocked FPs', ha='center', va='center', rotation='vertical')

    plt.show()

def drawBlockedThresholdScoreChart():
#Adjusted confidence test ratio(distance=0.50, neighbor_number=5, pos_dis=150)
    keys = [15,-15,30,-30,45,-45,180,False]
    x = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    lamp = [
                [1.00, 1.00, 1.00, 0.84, 0.78, 0.75, 0.72, 0.66, 0.63],
                [1.00, 1.00, 1.00, 0.93, 0.83, 0.72, 0.66, 0.64, 0.62],
                [1.00, 1.00, 1.00, 1.00, 0.98, 0.88, 0.82, 0.76, 0.75],
                [1.00, 1.00, 1.00, 0.96, 0.88, 0.81, 0.75, 0.70, 0.68],
                [1.00, 1.00, 0.90, 0.75, 0.72, 0.69, 0.67, 0.65, 0.63],
                [1.00, 1.00, 1.00, 0.72, 0.68, 0.65, 0.60, 0.58, 0.56],
                [0.20, 0.15, 0.12, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08],
                [0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02]
            ]
    horse = [
                [0.74, 0.74, 0.70, 0.60, 0.55, 0.49, 0.45, 0.42, 0.42],
                [0.79, 0.75, 0.74, 0.61, 0.58, 0.50, 0.49, 0.45, 0.44],
                [0.92, 0.84, 0.74, 0.71, 0.67, 0.62, 0.57, 0.52, 0.49],
                [0.47, 0.47, 0.44, 0.38, 0.36, 0.35, 0.31, 0.29, 0.26],
                [0.90, 0.73, 0.55, 0.51, 0.50, 0.49, 0.47, 0.44, 0.41],
                [0.34, 0.34, 0.33, 0.30, 0.23, 0.22, 0.20, 0.19, 0.19],
                [0.14, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
                [0.09, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05]
            ]
    pops = [
                [1.00, 1.00, 0.95, 0.85, 0.78, 0.74, 0.70, 0.63, 0.62],
                [1.00, 1.00, 1.00, 0.84, 0.81, 0.71, 0.66, 0.63, 0.60],
                [1.00, 1.00, 1.00, 1.00, 1.00, 0.92, 0.86, 0.81, 0.77],
                [0.29, 0.17, 0.14, 0.12, 0.12, 0.12, 0.12, 0.12, 0.11],
                [1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.86, 0.82, 0.77],
                [0.43, 0.35, 0.23, 0.19, 0.18, 0.17, 0.17, 0.16, 0.16],
                [0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                [0.49, 0.43, 0.36, 0.28, 0.26, 0.24, 0.23, 0.22, 0.21]
            ]

    fig, axs = plt.subplots(1,3, sharey=True)
    ds = [horse, lamp, pops]
    ts = ["horse", "lamp", "pops"]
    labels = [str(i) for i in x] 
    for i, d in enumerate(ds):
        for k, values in enumerate(ds[i]):
            axs[i].plot(x, values, label=str(keys[k]))
        axs[i].set_title(ts[i])
#    ax.plot(list(range(len(keys))), ave_score, label='score given by our algorithm')
    plt.legend(loc='best')
    fig.text(0.5, 0.04, 'ratio', ha='center', va='center')
    fig.text(0.06, 0.5, 'score', ha='center', va='center', rotation='vertical')

    plt.show()

if __name__ == "__main__":
#    drawFigures("horse")
#    drawPrecisionChart()
#    drawBlockedPointChart()
#    drawBlockedThresholdScoreChart()
