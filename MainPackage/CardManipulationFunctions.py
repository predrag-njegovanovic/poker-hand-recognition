import numpy as np
import cv2
import matplotlib.pyplot as plt
from operator import itemgetter



def haveSpace(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5, 5), np.float32) / 25

    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    i = turnWhite(img, i)

    for x in range(len(i)):
        clearColumn = True

        for k in range(len(i[x])):
            if (i[x][k] == 1):
                clearColumn = False
                break

        if(clearColumn):
            return True

    return  False


def getBlacks(i):
    whites = 0
    for x in range(len(i)):
        for k in range(len(i[x])):
            if (i[x][k] == 1):
                whites += 1
    return whites

def cutInHalf(img):
    height, width, channel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5, 5), np.float32) / 25
    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    bimg = turnWhite(img, i)
    #plt.imshow(bimg,'gray')
    #plt.show()
    firstColumn = bimg[:, 5:10]
    lastColumn = bimg[:, width-10:width-5]
    firstRow = bimg[5:10, :]
    lastRow = bimg[height-10:height-5, :]

    TopImage = img[0:height/2-20, 0:width]
    BottomImage = img[height/2+20:height, 0:width]
    LeftImage = img[0:height, 0:width/2+20]
    RightImage = img[0:height, width/2-20:width]

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
    res = im
    height, width, channel = im.shape
    if(width > height):
        res = cv2.resize(im, (120, 120), interpolation=cv2.INTER_CUBIC)
        Mat = cv2.getRotationMatrix2D((60, 60), -90, 1)
        im = cv2.warpAffine(res, Mat, (120, 120))
        res = cv2.resize(im, (height, width), interpolation=cv2.INTER_CUBIC)
    #plt.imshow(res)
    #plt.show()
    return res

def nadjiKonture (img):
    nesto, konture, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    x, y, w, h = nadjiKonture(crops)
    topOut = cropsTop[y:y+h, x:x+w]
    cropHeight, cropWidth, channel = cropsTop.shape
    height, width, channel = topOut.shape
    if(width > height):
        Mat = cv2.getRotationMatrix2D((cropWidth/2, cropHeight/2), angleRotate, 1)
        novaSlika = cv2.warpAffine(cropsTop, Mat, (cropWidth, cropHeight))
        cc = obradiSliku(novaSlika, counter)
        x,y,w,h = nadjiKonture(cc)
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

def turnWhite(img, blackImg):
    h, w, c = img.shape
    for x in range(h):
        for j in range(w):
            if((img[x][j][0] > 120 and img[x][j][1] < 80 and img[x][j][2] < 80) or (img[x][j][0] < 60 and img[x][j][1] < 60 and img[x][j][2] < 60)):
                blackImg[x][j] = 1

    return blackImg