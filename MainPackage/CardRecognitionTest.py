import matplotlib.pyplot as plt
from distutils.command.config import config

import numpy as np
from skimage.io import imread
import matplotlib.image as im
import cv2
import copy

img = imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
copy = img.copy()
filter = np.ones((5,5), np.float32)/25

i = cv2.filter2D(gray, -1, filter)
i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

kernel = np.ones((3, 3), np.uint8)

i = cv2.morphologyEx(i,cv2.MORPH_OPEN, kernel, iterations = 1)
i = cv2.dilate(i, kernel, iterations= 25)
ic = i.copy()
image, contours, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(copy, contours, -1, (255, 255, 0), 3)

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

print list[idx], max

x = list[idx][0]
y = list[idx][1]
w = list[idx][2]
h = list[idx][3]
#cv2.rectangle(copy, (x,y),(x+w,y+h),(255,255,0),3)

plt.imshow(copy)
plt.show()

