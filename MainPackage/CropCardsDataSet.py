import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import matplotlib.image as im
import cv2
import copy
import os
from PIL import Image
import CardManipulationFunctions as cm


allPictures = os.listdir('Soft-dataset')
for card in allPictures:
    img = imread('Soft-dataset/'+card)
    #img = imread('12 PIK.jpg')

    #
    counter = 7
    wFirstCounter = 0
    while wFirstCounter < 100:
        counter += 2
        i = cm.obradiSliku(img, counter)

        image, contours, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        c = contour[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        list = []
        for idx, i in enumerate(approx):
            list.append(approx[idx][0])

        wFirstCounter+=1
        if(len(approx) <= 4):
            break

    approx = np.array(list, np.float32)
    dst = np.array([[0, 0], [0, 499], [499, 499], [499, 0]], np.float32)
    M = cv2.getPerspectiveTransform(approx, dst)
    z = cv2.warpPerspective(img, M, (500, 500))
    cropsTop = z[0:120,0:120]
    cropsBottom = z[380:500, 0:120]
    angleRotate = -90
    secCounter = 2
    wFirstCounter = 0
    while wFirstCounter < 20:
        whites = 0
        blacks = 0
        tO = cm.Znak(cropsTop, secCounter, angleRotate)
        h, w, c = tO.shape
        gray = cv2.cvtColor(tO, cv2.COLOR_BGR2GRAY)
        filter = np.ones((5, 5), np.float32) / 25
        i = cv2.filter2D(gray, -1, filter)
        i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
        for x in range(len(i)):
            for k in range(len(i[x])):
                if(i[x][k] == 0):
                    blacks+=1
                else:
                    whites+=1

        wFirstCounter += 1
        secCounter += 1
        if(blacks/whites > 5):
            continue

        if(h > 80 and h < 110 and w > 40 and w < 100):
            break

    ### Crops bottom ###
    secCounter = 2
    wFirstCounter = 0
    angleRotate = 90
    while wFirstCounter < 20:
        whites = 0
        blacks = 0
        bO = cm.Znak(cropsBottom, secCounter, angleRotate)
        h, w, c = bO.shape
        gray = cv2.cvtColor(bO, cv2.COLOR_BGR2GRAY)
        filter = np.ones((5, 5), np.float32) / 25
        i = cv2.filter2D(gray, -1, filter)
        i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
        for x in range(len(i)):
            for k in range(len(i[x])):
                if (i[x][k] == 0):
                    blacks += 1
                else:
                    whites += 1

        wFirstCounter += 1
        secCounter += 1
        if (blacks / whites > 5):
            continue

        if (h > 80 and h < 110 and w > 40 and w < 100):
            break

    ht, wt, ct = tO.shape
    if(ht > 115 and wt > 115):
        tO = cm.cutInHalf(tO)

    hb, wb, cb = bO.shape
    if(hb > 115 and wb > 115):
        bO = cm.cutInHalf(bO)

    top = Image.fromarray(tO)
    bottom = Image.fromarray(bO)
    imageTop = top.resize((95,70), Image.ANTIALIAS)
    imageBottom = bottom.resize((95, 70), Image.ANTIALIAS)
    imageTop.save(os.path.join('Soft-dataset-uglovi',"Top "+card), 'JPEG', quality=90)
    imageBottom.save(os.path.join('Soft-dataset-uglovi',"Bottom "+card), 'JPEG', quality=90)




