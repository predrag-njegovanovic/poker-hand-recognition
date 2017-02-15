import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import matplotlib.image as im
import cv2
import copy
import os
from PIL import Image
from operator import itemgetter


def getBlacks(img):
    whites = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5, 5), np.float32) / 25
    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    for x in range(len(i)):
        for k in range(len(i[x])):
            if (i[x][k] == 1):
                whites += 1
    return whites

def cutInHalf(img):
    height, width, channel = img.shape
    #
    firstColumn = img[:, 3:5]
    lastColumn = img[:, width-5:width]
    firstRow = img[3:5, :]
    lastRow = img[height-5:height, :]

    TopImage = img[0:height/2+6, 0:width]
    BottomImage = img[height/2-6:height, 0:width]
    LeftImage = img[0:height, 0:width/2-6]
    RightImage = img[0:height, width/2+6:width]

    fC = getBlacks(firstColumn)
    lC = getBlacks(lastColumn)
    fR = getBlacks(firstRow)
    lR = getBlacks(lastRow)
    #print fC, lC, fR, lR
    # #
    temp = ((fC, RightImage),(lC,LeftImage), (fR, BottomImage), (lR, TopImage))
    #
    pom = sorted(temp, key=itemgetter(0))[-1]
    im = pom[1]
    height, width, channel = im.shape
    if(width > height):
        res = cv2.resize(im, (120, 120), interpolation=cv2.INTER_CUBIC)
        Mat = cv2.getRotationMatrix2D((60, 60), -90, 1)
        im = cv2.warpAffine(res, Mat, (120, 120))
        res = cv2.resize(im, (height, width), interpolation=cv2.INTER_CUBIC)
    #plt.imshow(res)
    #plt.show()
    return res

def nadjiKonture (img, cropsTop):
    konture, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(cropsTop, konture, -1, (255, 255, 0), 2)
    kontura = sorted(konture, key=cv2.contourArea, reverse=True)

    list = []

    t = cv2.boundingRect(kontura[0])
    list.append(t)

    #print len(list)
    #cv2.rectangle(cropsTop, (t[0],t[1]),(t[1]+t[3],t[0]+t[2]), (255, 255, 0), 2)

    max = 0
    idx = 0
    for index, i in enumerate(list):
        if (i[3] > max):
            max = i[3]
            idx = index

    x = list[idx][0]
    y = list[idx][1]
    w = list[idx][2]
    h = list[idx][3]
    return x, y, w, h

def Znak(cropsTop, counter, angleRotate):
    crops = obradiSliku(cropsTop, counter)
    x, y, w, h = nadjiKonture(crops, cropsTop)
    topOut = cropsTop[y:y+h, x:x+w]
    cropHeight, cropWidth, channel = cropsTop.shape
    height, width, channel = topOut.shape
    if(width > height):
        Mat = cv2.getRotationMatrix2D((cropWidth/2, cropHeight/2), angleRotate, 1)
        novaSlika = cv2.warpAffine(cropsTop, Mat, (cropWidth, cropHeight))
        cc = obradiSliku(novaSlika, counter)
        x,y,w,h = nadjiKonture(cc, cropsTop)
        cv2.rectangle(cc, (x, y), (x + w, y + h), (255, 255, 0), 2)
        topOut = novaSlika[y:y+h, x:x+w]
        #plt.imshow(topOut)
        #plt.show()

    #cv2.rectangle(cropsTop,(x,y),(x+w,y+h,(255,255,0),2)

    return topOut

def obradiSliku (img, nmbOfDilate):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5, 5), np.float32) / 25
    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    kernel = np.ones((3, 3), np.uint8)

    i = cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel, iterations=1)
    i = cv2.dilate(i, kernel, iterations=nmbOfDilate)
    return i

allPictures = os.listdir('Soft-dataset')
for card in allPictures:
    img = imread('Soft-dataset/'+card)
    #img = imread('12 PIK.jpg')

    #
    counter = 7
    wFirstCounter = 0
    while wFirstCounter < 100:
        counter += 2
        i = obradiSliku(img, counter)

        contours, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        tO = Znak(cropsTop, secCounter, angleRotate)
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
        bO = Znak(cropsBottom, secCounter, angleRotate)
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
        tO = cutInHalf(tO)

    hb, wb, cb = bO.shape
    if(hb > 115 and wb > 115):
        bO = cutInHalf(bO)

    top = Image.fromarray(tO)
    bottom = Image.fromarray(bO)
    imageTop = top.resize((95,70), Image.ANTIALIAS)
    imageBottom = bottom.resize((95, 70), Image.ANTIALIAS)
    imageTop.save(os.path.join('Soft-dataset-uglovi',"Top "+card), 'JPEG', quality=90)
    imageBottom.save(os.path.join('Soft-dataset-uglovi',"Bottom "+card), 'JPEG', quality=90)




