import os
import re
import cv2
import numpy as np

from skimage.io import imread
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers.core import Activation, Dense, Dropout


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:, 0], ll[:,  1]] = 1
    return retVal


def trainNetwork():
    allPictures = os.listdir('Soft-dataset-duplicated')
    allPictures = natural_sort(allPictures)
    data = []
    labels = []
    for card in allPictures:
        card_number = int(card.split(' ', 2)[0])
        labels.append(card_number)
        img = imread('Soft-dataset-duplicated/' + card)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
        i = np.asarray(img.flatten())
        data.append(i)
    test_labels = np_utils.to_categorical(labels, 19)
    data = np.array(data) / 255.0
    print(data.shape)

    # Defining a model
    model = Sequential()
    model.add(Dense(3000, input_dim=3325))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(750, activation='relu'))
    model.add(Dense(320, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(19))
    model.add(Activation("softmax"))
    adagrad = Adagrad(lr=0.0001, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])
    print("....Training starting....")

    training = model.fit(data, test_labels, nb_epoch=25, batch_size=10, verbose=1)
    print(training.history)
    print("...Training finished...")
    return model


def checkCard(model, testImg):
    g = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
    testImg = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    testX = np.asarray(testImg.flatten())
    testX = np.reshape(testX, (1, 3325))
    testX = testX/255.0
    t = model.predict(testX, verbose=0)
    maxIndex = np.argmax(t)
    return maxIndex, t[0][maxIndex]
