import matplotlib.pyplot as plt
import numpy as np
from numpy import random, exp, dot
from skimage.io import imread
import matplotlib.image as im
import cv2
import copy
import os
from decimal import Decimal
from PIL import Image
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

from sympy.logic.inference import pl_true


class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((2624,1)) - 1


    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


    def __sigmoid_derivative(self,x):
        return x * (1 - x)


    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output = self.predict(training_set_inputs)
            error = training_set_outputs - output
            adjusment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjusment


    def predict(self,inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))



if __name__ == "__main__":
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

        img = imread('Soft-dataset sredjen/' + card, 'r')
        pom.append(img.flatten())

    card_input = np.asarray(pom)

    card_number_targets = np.asarray(card_number_targets)
    matricaX = card_number_targets.reshape(len(card_number_targets),1)

    for index, x in enumerate(card_number_targets):
        matricaX[index][0] = x


    neural_network = NeuralNetwork()
    neural_network.train(card_input, matricaX, 100000)


    testImg = imread("kectTrefTest.jpg",'r')
    testX = np.asarray(testImg.flatten())
    print neural_network.predict(testX)











