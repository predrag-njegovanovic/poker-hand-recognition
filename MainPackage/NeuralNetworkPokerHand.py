from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import SGD,Adagrad



def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:, 0],ll[:,  1]] = 1
    return retVal

def trainPokerNetwork():
    f = open('Soft-dataset-pokerHand/PokerHandDataset.txt','r')
    textLines = []
    labels = []
    ranks = []
    numbers = []
    for line in f:
        newData = []
        parts = line.split(",")
        for i in xrange(0, len(parts), 2):
            if(i == 10):
                labels.append(int(parts[i]))
                break

            ranks.append(int(parts[i]))
            numbers.append(int(parts[i+1]))
        #textLines.append(newData)
    ranks = to_categorical(ranks, 19)
    numbers = to_categorical(numbers, 14)

    for x in range(len(ranks)):
        r = ranks[x,15:19]
        n = numbers[x,:]
        pom = []
        for t in range(len(n)):
            pom.append(n[t])
        for t in range(len(r)):
            pom.append(r[t])
        textLines.append(pom)


    print "Priprema ulaza....."
    #
    data = []
    for x in  textLines:
        data.append(x)
    #
    data = np.array(data)
    #
    print "Zavrsena priprema....."
    #
    testLabels = to_categorical(labels,10)
    #
    # model = Sequential()
    # model.add(Dense(5, input_dim=10))
    # model.add(Activation('relu'))
    # model.add(Dense(10, activation='softmax'))
    # sgd = SGD(lr=0.001, momentum=0.75, decay=1e-06, nesterov=True)
    # adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #
    # print "...Training starting..."
    #
    # training = model.fit(data,testLabels,batch_size=100,nb_epoch=10,verbose=1)
    #
    # print training.history
    # print "...Training finished...."
    #
    # return model

def getPokerHand(model, array):
    t = model.predict(array, verbose=0)
    maxIndex = np.argmax(t)
    return maxIndex, t


trainPokerNetwork()