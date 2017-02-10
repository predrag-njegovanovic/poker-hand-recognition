import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import matplotlib.image as im
import cv2
import copy
import os
from PIL import Image
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from sklearn.datasets import fetch_mldata

from sympy.logic.inference import pl_true

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

allPictures = os.listdir('Soft-dataset sredjen')

# [card_number, card_symbol]
# herc = 1, karo = 2, pik = 3, tref = 4
card_targets = []
card_input = [] # card_pixels

#za test
card_number_targets = []
card_symbol_targets = []
pom = []
#-======

for card in allPictures:
    card_number = int(card.split(' ', 2)[0])
    card_symbol = card.split(' ', 2)[1].split(".", 2)[0]

    if(card_symbol == "HERC"):
        card_symbol = 1
    elif(card_symbol == "KARO"):
        card_symbol = 2
    elif(card_symbol == "PIK"):
        card_symbol = 3
    elif(card_symbol == "TREF"):
        card_symbol = 4
    else:
        print ("Greska prilikom uzimanja symbola!")

    card_targets.append([card_number, card_symbol])

    #test
    card_number_targets.append(card_number)
    card_symbol_targets.append(card_symbol_targets)
    #--

   # img = Image.open('Soft-dataset sredjen/' + card, 'r')
 #   pix_val = list(img.getdata())
 #   card_input.append(pix_val)

    img = imread('Soft-dataset sredjen/' + card, 'r')
    pom.append(img.flatten())

card_input = np.asarray(pom)
test_numbers_out = to_categorical(card_number_targets, 52)



#    profesorovo

#    img = imread('Soft-dataset sredjen/' + card, 'r')
#    pom.append(img.flatten())

# #
# #card_input = np.asarray(pom)
#
# card_input = np.asarray(pom)
# test_numbers_out = to_categorical(card_number_targets, 52)
#
# #model
#
# model = Sequential()
# model.add(Dense(70, input_dim=2624))
# model.add(Activation("relu"))
# model.add(Dense(52))
# model.add(Activation("relu"))
#
# sgd = SGD(0.1, 0.7,0.001)
# model.compile(loss='mean_squared_error', optimizer=sgd)
#
# training = model.fit(card_input,test_numbers_out, nb_epoch=500, batch_size=400, verbose=0)
# print training.history['loss'][-1]
#
#
# testImg = imread("kectTrefTest.jpg",'r')
#
#
# testX = np.asarray(testImg.flatten())
#
# testX = np.reshape(testX, testX.shape + (1,))
#
# t = model.predict(testX, verbose=1)
# #print t[testImg]

