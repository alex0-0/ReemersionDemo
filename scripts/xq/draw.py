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
    labels = ['template',15,-15,30,-30,45,-45,180,'false']
    for i, img in enumerate(images):
        ax = f.add_subplot(3, 3, i+1)
        ax.set_title(str(labels[i]))#+'$^\circ$')
        plt.axis('off')
        plt.imshow(img)
    plt.subplots_adjust(left=0.125, right=0.6, top=0.90, bottom=0.51, hspace=0.30, wspace=0.00)
    plt.show()

def drawPrecisionChart(ds=['horse','lamp','GrandfatherClock','pops','DishSoap','Motorcycle','Robot']):
    fnames = []
    fnames.append("000.JPG")
    fnames.append("015.JPG")
    fnames.append("-15.JPG")
    fnames.append("030.JPG")
    fnames.append("-30.JPG")
    fnames.append("045.JPG")
    fnames.append("-45.JPG")
#    fnames.append("180.JPG")
    fnames.append("false.JPG")

    keys = [15,-15,30,-30,45,-45,-9999999]

#    def readImages(d):
#        r = []
#        for f in fnames:
#            r.append(plt.imread(d+"/"+f))
#        return r
#
#    SURF = []
#    ORB = []
#    images = []
#    for d in ds:
#        images.append(readImages(d))
#    for ig in images:
#        SURF.append([test.testMatchPrecision(i, ig[0],detect_method=detect.extractSURFFeatures) for i in ig[1:]])
#        ORB.append([test.testMatchPrecision(i, ig[0],detect_method=detect.extractORBFeatures) for i in ig[1:]])
#
#    if DEGBUG > 1:
#        print(TAG + "SURF precision list: " + str(SURF))
#        print(TAG + "ORB precision list: " + str(ORB))
#    
#    ave_SURF = []
#    ave_ORB = []
#    for i, t in enumerate(SURF[0]):
#        ave_SURF.append(sum([p[i] for p in SURF])/len(SURF))
#        ave_ORB.append(sum([p[i] for p in ORB])/len(ORB))
#    print(ave_SURF)
#    print(ave_ORB)
#    exit()

#negative values come from test.testAlgorithmPrecision. Everything else come from above code
    ave_SURF = [0.1692262774227666, 0.14014296365455794, 0.10426864040749992, 0.07682533589403819, 0.06287563807907229, 0.05127279996421956, 0.02804886157940623]
    ave_ORB = [0.15257142857142858, 0.1522857142857143, 0.07114285714285715, 0.053714285714285714, 0.027142857142857142, 0.023714285714285712, 0.004489795918367348]

    dic_SURF = dict(zip(keys, ave_SURF))
    dic_ORB = dict(zip(keys, ave_ORB))
#    dic = {15: 0.167, -15: 0.098, 30: 0.107, -30: 0.007, 45: 0.030, -45: 0.002, 180: 0.002, False: 0.002}
    k1 = sorted(dic_SURF)
#    k2 = sorted(dic_ORB)
    v1 = [dic_SURF[k] for k in k1]
    v2 = [dic_ORB[k] for k in k1]
    labels = [str(k) for k in k1]
    labels[0] = "negative"

    #copied from tests
    #Adjusted confidence test number of matches(ratio=0.50, neighbor_number=5, pos_dis=150) ratio=0.5
    #horse = np.array([0.64, 0.44, 0.34, 0.19, 0.29, 0.24, 0.07, 0.04])
    #lamp = np.array([0.8, 0.83, 0.97, 1, 1, 0.97, 0.17, 0.03])
    #pops = np.array([0.86, 0.87, 1, 0.41, 1, 0.45, 0.12, 0.33])
    #grandfater_clock = np.array([0.77, 0.78, 0.97, 0.99, 1.00, 1.00, 0.00, 0.22])
    #motocycle = np.array([0.78, 0.79, 0.66, 0.63, 0.96, 0.92, 0.0, 0.71])
    #robot = np.array([0.84, 0.82, 1, 1, 1, 1, 0.0, 0.4])
    #dishsoap = np.array([0.83, 0.83, 1, 0.98, 1, 1, 0.0, 0.06])
    horse = np.array([0.64, 0.44, 0.34, 0.19, 0.29, 0.24])
    lamp = np.array([0.8, 0.83, 0.97, 1, 1, 0.97])
    pops = np.array([0.86, 0.87, 1, 0.41, 1, 0.45])
    grandfater_clock = np.array([0.77, 0.78, 0.97, 0.99, 1.00, 1.00])
    motocycle = np.array([0.78, 0.79, 0.66, 0.63, 0.96, 0.92])
    robot = np.array([0.84, 0.82, 1, 1, 1, 1])
    dishsoap = np.array([0.83, 0.83, 1, 0.98, 1, 1])
    
#    ave_score = (horse+lamp+grandfater_clock)/len(ds)
    ave_score = (horse+lamp+pops+grandfater_clock+motocycle+robot+dishsoap)/7
    ave_score = np.append(ave_score, 0.3744810294833636)
#    print(ave_score)
#    exit()
#    ave_score = [ 0.78857143, 0.76571429, 0.84857143, 0.74285714, 0.89285714, 0.79714286, 0.25571429]
    dic = dict(zip(keys, ave_score))
    ave_score = [dic[k] for k in k1]

    """
    In pyplot drawing function, the x stands for how x axis should be seperated, it it not necessary to use real key as x value
    To make the spacing equal, we can use list(range(len(keys))) to be the x value and then change the x labels.
    """
    fig, ax = plt.subplots(1,1)
    ax.plot(list(range(len(keys))), v1, label='SURF', marker='s')
    ax.plot(list(range(len(keys))), v2, label='ORB', marker='o')
    ax.plot(list(range(len(keys))), ave_score, label='FARES', marker='+')
    ax.set_xticklabels([""]+labels)
    plt.xlabel("images")
    plt.ylabel("score")
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
    keys = [15,-15,30,-30,45,-45,180,"false"]
    x = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    lamp = [
                [244, 230, 193, 126, 109, 78 , 45 , 21 , 13],
                [242, 222, 186, 138, 109, 89 , 65 , 35 , 11],
                [250, 245, 204, 142, 115, 96 , 50 , 19 , 12],
                [247, 242, 208, 137, 117, 88 , 61 , 26 , 16],
                [270, 256, 230, 156, 108, 74 , 55 , 24 , 14],
                [272, 266, 217, 138, 116, 101, 55 , 27 , 15],
                [268, 251, 191, 137, 120, 97 , 46 , 32 , 10],
                [286, 200, 147, 83 , 75 , 54 , 28 , 18 , 9 ]
            ]
    horse = [
                [250, 244, 211, 140, 116, 90 , 56 , 30 , 9 ],
                [250, 242, 214, 153, 122, 75 , 49 , 22 , 11],
                [265, 257, 213, 153, 112, 75 , 46 , 25 , 9 ],
                [257, 238, 212, 159, 127, 93 , 61 , 25 , 10],
                [267, 264, 222, 126, 104, 70 , 48 , 18 , 8 ],
                [250, 249, 222, 163, 129, 105, 54 , 21 , 12],
                [281, 280, 226, 145, 105, 51 , 23 , 9  , 7 ],
                [287, 271, 203, 118, 86 , 71 , 36 , 22 , 9 ]
            ]
    clock = [
                [220, 206, 180, 129, 108, 83 , 65 , 30 , 13],
                [247, 219, 167, 134, 113, 87 , 64 , 25 , 10],
                [246, 234, 193, 147, 114, 88 , 55 , 27 , 9 ],
                [216, 206, 187, 136, 122, 97 , 54 , 24 , 12],
                [250, 245, 205, 146, 119, 98 , 61 , 21 , 7 ],
                [235, 203, 170, 133, 110, 87 , 64 , 23 , 10],
                [19 , 19 , 19 , 19 , 19 , 19 , 19 , 19 , 19],
                [274, 261, 222, 147, 114, 92 , 50 , 23 , 13]
            ]

    fig, axs = plt.subplots(1,3, sharey=True)
    ds = [horse, lamp, clock]
    ts = ["horse", "lamp", "clock"]
    labels = [str(i) for i in x]
#    linestyles = ['--', '-.', '-', ':', (0,(5,1)), (0,(3,5,1,5)), (0,(5,5)), (0,(3,1,1,1))]
    
    for i, d in enumerate(ds):
        for k, values in enumerate(ds[i]):
            if keys[k] == 180 or keys[k] == "false":
                linestyle = '--'
            else:
                linestyle = '-'
            axs[i].plot(x, values, label=str(keys[k]), linestyle = linestyle)
        axs[i].set_title(ts[i])
#    ax.plot(list(range(len(keys))), ave_score, label='score given by our algorithm')
    plt.legend(loc='best')
    fig.text(0.5, 0.04, '$T_b$', ha='center', va='center')
    fig.text(0.06, 0.5, 'number of blocked FPs', ha='center', va='center', rotation='vertical')

    plt.show()

def drawBlockedThresholdScoreChart():
#Adjusted confidence test ratio(distance=0.50, neighbor_number=5, pos_dis=150)
    keys = [15,-15,30,-30,45,-45,180,"false"]
    x = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    lamp = [
                [1.00, 1.00, 0.98, 0.80, 0.77, 0.71, 0.66, 0.63, 0.62],
                [1.00, 1.00, 0.96, 0.83, 0.77, 0.73, 0.69, 0.65, 0.61],
                [1.00, 1.00, 1.00, 0.97, 0.90, 0.86, 0.77, 0.72, 0.71],
                [1.00, 1.00, 1.00, 1.00, 0.97, 0.90, 0.84, 0.78, 0.77],
                [1.00, 1.00, 1.00, 0.85, 0.75, 0.69, 0.66, 0.62, 0.60],
                [1.00, 1.00, 0.93, 0.72, 0.68, 0.66, 0.59, 0.55, 0.54],
                [0.26, 0.24, 0.19, 0.17, 0.16, 0.15, 0.13, 0.13, 0.12],
                [0.07, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
            ]
    horse = [
                [0.92, 0.90, 0.80, 0.64, 0.60, 0.56, 0.52, 0.49, 0.47],
                [0.61, 0.59, 0.53, 0.44, 0.40, 0.36, 0.34, 0.32, 0.31],
                [0.51, 0.49, 0.41, 0.34, 0.31, 0.28, 0.26, 0.25, 0.24],
                [0.26, 0.24, 0.22, 0.19, 0.17, 0.16, 0.15, 0.14, 0.13],
                [0.35, 0.35, 0.29, 0.22, 0.21, 0.19, 0.18, 0.17, 0.17],
                [0.25, 0.24, 0.22, 0.18, 0.17, 0.16, 0.14, 0.13, 0.13],
                [0.12, 0.12, 0.10, 0.07, 0.07, 0.06, 0.05, 0.05, 0.05],
                [0.08, 0.07, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03]
            ]
    clock = [
                [1.00, 1.00, 0.94, 0.81, 0.77, 0.72, 0.69, 0.64, 0.62],
                [1.00, 1.00, 0.90, 0.82, 0.78, 0.73, 0.69, 0.63, 0.61],
                [1.00, 1.00, 1.00, 1.00, 0.97, 0.91, 0.84, 0.79, 0.76],
                [1.00, 1.00, 1.00, 1.00, 0.99, 0.93, 0.84, 0.79, 0.77],
                [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                [0.38, 0.36, 0.31, 0.24, 0.22, 0.21, 0.19, 0.18, 0.17]
            ]

    fig, axs = plt.subplots(1,3, sharey=True)
    ds = [lamp, clock, horse]
    ts = ["lamp", "clock", "horse"]
    labels = [str(i) for i in x]
#    linestyles = ['--', '-.', '-', ':', (0,(5,1)), (0,(3,5,1,5)), (0,(5,5)), (0,(3,1,1,1))]
    for i, d in enumerate(ds):
        for k, values in enumerate(ds[i]):
            if keys[k] == 180 or keys[k] == "false":
                linestyle = '--'
            else:
                linestyle = '-'
            axs[i].plot(x, values, label=str(keys[k]), linestyle = linestyle)
        axs[i].set_title(ts[i])
#    ax.plot(list(range(len(keys))), ave_score, label='score given by our algorithm')
    plt.legend(loc='upper right', prop={'size':8})
#    plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=4)

#    plt.legend(bbox_to_anchor=(-0.1,1.02,1,0.2), loc="lower left",
#                mode="expand", borderaxespad=0, ncol=2)
#    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", prop={'size':7},)

#    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=8)
    fig.text(0.5, 0.04, '$T_b$', ha='center', va='center')
    fig.text(0.06, 0.5, 'score', ha='center', va='center', rotation='vertical')

    plt.show()
#    fig.savefig('ratio_score', bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == "__main__":
    drawFigures("horse")
#    drawPrecisionChart()
#    drawBlockedPointChart()
#    drawBlockedThresholdScoreChart()
