import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import matplotlib.image as im
import cv2
import copy
import os
import re
from PIL import Image
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import operator


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:, 0],ll[:,  1]] = 1
    return retVal

allPictures = os.listdir('Soft-dataset sredjen')
allPictures = natural_sort(allPictures)
# [card_number, card_symbol]
# herc = 1, karo = 2, pik = 3, tref = 4
data = []
labels = []
card_symbol = []

for card in allPictures:
    card_number = int(card.split(' ', 2)[0])
    symbol = card.split(' ', 2)[1].split(".", 2)[0]

    if(symbol == "HERC"):
        symbol = 1
    elif(symbol == "KARO"):
        symbol = 2
    elif(symbol == "PIK"):
        symbol = 3
    elif(symbol == "TREF"):
        symbol = 4
    else:
        print ("Greska prilikom uzimanja symbola!")

    labels.append(card_number)
    card_symbol.append(symbol)


    img = imread('Soft-dataset sredjen/' + card)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    i = np.asarray(img.flatten())
    data.append(i)


test_labels = np_utils.to_categorical(labels, 15)
#ll = np.asarray(card_symbol)
#test_labels[:, 0] = ll
#print test_labels


data = np.array(data) / 255.0
#print data
# # #model
# #
model = Sequential()
model.add(Dense(1312, input_dim=2624))
model.add(Activation("relu"))
model.add(Dense(768, activation='relu'))
model.add(Dense(768, activation='relu'))
model.add(Dense(334, activation='relu'))
model.add(Dense(15))
model.add(Activation("softmax"))
sgd = SGD(0.1, 0.75, 0.001)
model.compile(loss='mean_squared_error', optimizer=sgd)
#

print "....Training starting...."

training = model.fit(data,test_labels, nb_epoch=10, batch_size=5, verbose=0)
print training.history
print "...Training finished..."


# #
testImg = imread("test10pik.jpg")
# # #
# # #
g = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
testImg = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
testX = np.asarray(testImg.flatten())
testX = np.reshape(testX, (1, 2624))
testX = testX/255.0
t = model.predict(testX, verbose=1)
print t
maxIndex = np.argmax(t)
print maxIndex
