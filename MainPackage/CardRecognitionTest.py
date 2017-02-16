import matplotlib.pyplot as plt

import numpy as np
from skimage.io import imread
import cv2
from skimage.measure import label
from skimage.measure import regionprops
import copy
import NeuralNetwork as NN
from PIL import Image

def Konture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5,5), np.float32)/25
    copy = img.copy()
    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    kernel = np.ones((3, 3), np.uint8)

    i = cv2.morphologyEx(i,cv2.MORPH_OPEN, kernel, iterations = 1)
    i = cv2.dilate(i, kernel, iterations = 3)
    contours, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    #cv2.drawContours(copy, contours, -1, (255, 255, 0), 3)

    ####Ovo je hardcodovano
    card = contours[0]
    peri = cv2.arcLength(card, True)
    approx = cv2.approxPolyDP(card, 0.02*peri, True)
    rect = cv2.minAreaRect(card)
    r = cv2.cv.BoxPoints(rect)

    list = []

    for k in contours:
        t = cv2.boundingRect(k)
        list.append(t)


    max = 0
    idx = 0
    for index, i in enumerate(list):
        if(i[3] > max):
            max = i[3]
            idx = index

    #print list[idx], max

    x = list[idx][0]
    y = list[idx][1]
    w = list[idx][2]
    h = list[idx][3]
    #cv2.rectangle(copy, (x,y),(x+w,y+h),(255,255,0),3)
    copy = img[y: y + h, x: x + w]
    return approx

img = imread('Proba.jpg')
approx = Konture(img)

list = []
for idx, i in enumerate(approx):
    list.append(approx[idx][0])

approx = np.array(list, np.float32)
dst = np.array([[30, 30], [30, 499], [499, 499], [499, 30]],np.float32)
M = cv2.getPerspectiveTransform(approx, dst)
z = cv2.warpPerspective(img, M, (530, 530))

z = Image.fromarray(z)
image = z.resize((41, 64), Image.ANTIALIAS)
#image.save('slika','JPEG', quality = 90)
testCard = np.array(image)
r = imread('4 TREF.jpg')
model = NN.trainNetwork()
NN.checkCard(model, r)

plt.imshow(testCard)
plt.show()

