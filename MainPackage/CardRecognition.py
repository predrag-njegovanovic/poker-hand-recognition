import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import CardManipulationFunctions as cm
import cv2
import copy
import NeuralNetwork as NN


img = imread('KosaTest1.jpg')
c = copy.copy(img)
counter = 0
wFirstCounter = 0
arrayOfImages = []

while wFirstCounter < 100:
    counter += 1
    i = cm.obradiSliku(img, counter)
    konture, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(c, konture, -1, (255, 255, 0), 2)
    kontura = sorted(konture, key=cv2.contourArea, reverse=True)[:8]
    x = 0
    j = 0
    listOfContours = []
    listOfContours.append(kontura[0])
    while x < len(kontura):
        f = cv2.moments(kontura[x])
        cX = int(f["m10"]/f["m00"])
        cY = int(f["m01"]/f["m00"])
        flag = False
        j = 0
        while j < len(listOfContours):
            v = cv2.moments(listOfContours[j])
            vX = int(v["m10"] / v["m00"])
            vY = int(v["m01"] / v["m00"])
            j += 1
            if(abs(vX - cX) >= 0 and abs(vX - cX) < 5 and abs(vY - cY) >= 0 and abs(vY - cY) < 5):
                flag = True
                break
        if(flag == False):
            listOfContours.append(kontura[x])
            if(len(listOfContours) == 6):
                break
        x+=1

    flag = False
    arrayOfContures = []

    for idx,c in enumerate(listOfContours):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if(len(approx) > 4):
            flag = True
        list = []
        for idx, i in enumerate(approx):
            list.append(approx[idx][0])

        arrayOfContures.append(list)
    wFirstCounter += 1
    if (flag == False):
        break

for list in arrayOfContures:
    approx = np.array(list, np.float32)
    dst = np.array([[0, 0], [0, 499], [499, 499], [499, 0]], np.float32)
    M = cv2.getPerspectiveTransform(approx, dst)
    z = cv2.warpPerspective(img, M, (500, 500))
    #plt.imshow(z)
    #plt.show()
    arrayOfImages.append(z)

cropCards = []
for image in arrayOfImages:
     crops = image[0:120,0:120]
     secCounter = 2
     wFirstCounter = 0
     angleRotate = -90
     while wFirstCounter < 20:
         whites = 0
         blacks = 0
         tO = cm.Znak(crops, secCounter, angleRotate)
         #plt.imshow(tO)
         #plt.show()
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
#
         if(h > 80 and h < 110 and w > 40 and w < 100):
             break
#
     ht, wt, ct = tO.shape
     if (ht > 115 and wt > 115):
         tO = cm.cutInHalf(tO)


     cropCards.append(tO)

resizes = []
for card in cropCards:
    s = cv2.resize(card,(95, 70), interpolation=cv2.INTER_CUBIC)
    resizes.append(s)
    plt.imshow(card)
    plt.show()

#model = NN.trainNetwork()
#for r in resizes:
    #NN.checkCard(model, r)