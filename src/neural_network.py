import os
import re
import cv2
import numpy as np

from skimage.io import imread
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers.core import Activation, Dense, Dropout

full_path = os.path.dirname(__file__)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:, 0], ll[:,  1]] = 1
    return retVal


def train_network():
    pictures = os.listdir(full_path + '/data/processed_cards/')
    sorted_pictures = natural_sort(pictures)
    data = []
    labels = []
    for card in sorted_pictures:
        card_number = int(card.split(' ', 2)[0])
        labels.append(card_number)
        img = imread(full_path + '/data/processed_cards/' + card)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(gray,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    11,
                                    3)
        i = np.asarray(img.flatten())
        data.append(i)
    test_labels = np_utils.to_categorical(labels, 19)
    data = np.array(data) / 255.0
    # print(data.shape)

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

    training = model.fit(data,
                         test_labels,
                         nb_epoch=20,
                         batch_size=10,
                         verbose=1)
    print(training.history)
    print("...Training finished...")
    return model


def check_card(model, test_img):
    g = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_img = cv2.adaptiveThreshold(g,
                                     255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     11,
                                     3)
    test_x = np.asarray(test_img.flatten())
    test_x = np.reshape(test_x, (1, 3325))
    test_x = test_x/255.0
    t = model.predict(test_x, verbose=0)
    max_index = np.argmax(t)
    return max_index, t[0][max_index]
