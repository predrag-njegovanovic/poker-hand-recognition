import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import matplotlib.image as im
import cv2
import copy
import os
from PIL import Image
from operator import itemgetter


def Znak(cropsTop):
    crops = obradiSliku(cropsTop)

    def nadjiKonture(img):
        konture, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        list = []

        for k in konture:
            t = cv2.boundingRect(k)
            list.append(t)

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
        return x,y,w,h

    x, y, w, h = nadjiKonture(crops)
    topOut = cropsTop[y:y+h, x:x+w]
    cropHeight, cropWidth, channel = cropsTop.shape
    height, width, channel = topOut.shape
    if(width >= height):
        Mat = cv2.getRotationMatrix2D((cropWidth/2, cropHeight/2), -90, 1)
        novaSlika = cv2.warpAffine(cropsTop, Mat, (cropWidth, cropHeight))
        cc = obradiSliku(novaSlika)
        x,y,w,h = nadjiKonture(cc)
        #cv2.rectangle(cc, (x, y), (x + w, y + h), (255, 255, 0), 2)
        #plt.imshow(novaSlika)
        #plt.show()
        topOut = novaSlika[y:y+h, x:x+w]

    if(width > 100):
        topOut = topOut[0:height-10, 50:width]
    #cv2.rectangle(cropsTop,(x,y),(x+w,y+h,(255,255,0),2)

    return topOut



allPictures = os.listdir('Soft-dataset')
#for card in allPictures:
#img = imread('Soft-dataset/'+card)
#if(card.startswith("1 ")):
#    continue
img = imread('1 HERC.jpg')
def obradiSliku(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5, 5), np.float32)/25
    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    kernel = np.ones((3, 3), np.uint8)

    i = cv2.morphologyEx(i,cv2.MORPH_OPEN, kernel, iterations = 1)
    i = cv2.dilate(i, kernel, iterations= 13)
    return i

i = obradiSliku(img)

contours, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(copy,contours, -1, (255, 255, 0), 3)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

c = contours[0]
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.02 * peri, True)
list = []
for idx, i in enumerate(approx):
    list.append(approx[idx][0])


approx = np.array(list, np.float32)
dst = np.array([[0, 0], [0, 499], [499, 499], [499, 0]], np.float32)
M = cv2.getPerspectiveTransform(approx, dst)
z = cv2.warpPerspective(img, M, (500, 500))
cropsTop = z[0:120,0:120]
cropsBottom = z[390:485, 15:75]
tO = Znak(cropsTop)
plt.imshow(tO)
plt.show()
#bO = Znak(cropsBottom)
#top = Image.fromarray(cropsTop)
#bottom = Image.fromarray(cropsBottom)
#top.save(os.path.join('Soft-dataset-uglovi',"Top "+card), 'JPEG', quality=90)
#bottom.save(os.path.join('Soft-dataset-uglovi',"Bottom "+card), 'JPEG', quality=90)




