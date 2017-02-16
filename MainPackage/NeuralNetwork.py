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
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers import Adagrad
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


def trainNetwork():
    allPictures = os.listdir('Soft-dataset-uglovi')
    allPictures = natural_sort(allPictures)
    #print allPictures
    # # [card_number, card_symbol]
    # # herc = 1, karo = 2, pik = 3, tref = 4
    data = []
    labels = []
    card_symbol = []
    #
    for card in allPictures:
        card_number = int(card.split(' ', 2)[1])
        symbol = card.split(' ', 2)[2].split(".", 2)[0]
    #
        if(symbol == "HERC"):
            symbol = 100
        elif(symbol == "KARO"):
            symbol = 200
        elif(symbol == "PIK"):
            symbol = 300
        elif(symbol == "TREF"):
            symbol = 400
        else:
            print ("Greska prilikom uzimanja symbola!")
    #
        labels.append(card_number)
    #
        img = imread('Soft-dataset-uglovi/' + card)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
        i = np.asarray(img.flatten())
        data.append(i)
    #
    #
    #print len(labels)
    test_labels = np_utils.to_categorical(labels, 15)
    ll = np.asarray(card_symbol)
    # #test_labels[:, 0] = ll
    #print test_labels
    #
    #
    data = np.array(data) / 255.0
    print data.shape
    # # # #model
    # # #
    model = Sequential()
    model.add(Dense(3000, input_dim=6650))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(750, activation='relu'))
    model.add(Dense(320, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(15))
    model.add(Activation("softmax"))
    adagrad = Adagrad(lr=0.0001, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)
    # #
    #
    print "....Training starting...."
    #
    training = model.fit(data,test_labels, nb_epoch=11, batch_size=10, verbose=1)
    print training.history
    print "...Training finished..."
    return model

def checkCard(model, testImg):

    # # #
    # # #
    g = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
    testImg = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    #plt.imshow(testImg, 'gray')
    #plt.show()
    testX = np.asarray(testImg.flatten())
    testX = np.reshape(testX, (1, 6650))
    testX = testX/255.0
    t = model.predict(testX, verbose=1)
    print t
    maxIndex = np.argmax(t)
    print maxIndex


# model = trainNetwork()
# testImg1 = imread("Bottom 2 HERC.jpg")
# testImg2 = imread("Bottom 13 KARO.jpg")
# testImg3 = imread("Top 3 PIK.jpg")
# checkCard(model, testImg1)
# checkCard(model, testImg2)
# checkCard(model, testImg3)