from __future__ import division
import numpy as np
import pickle as pick
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import SGD,Adagrad
import h5py


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:, 0],ll[:,  1]] = 1
    return retVal

def trainPokerNetwork():
    f = open('Soft-dataset-pokerHand/Test.txt','r')
    k = 1
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

            ranks = to_categorical([int(parts[i])], 19)
            numbers = to_categorical([int(parts[i+1])], 14)

            r = ranks[0, 15:19]
            n = numbers[0, :]

            newData.extend(r)
            newData.extend(n)

        k += 1
        print k
        textLines.append(newData)


    # with open("TextLines.txt", "wb") as fp:
    #     pick.dump(textLines, fp)

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

    model = Sequential()
    model.add(Dense(17, input_dim=90))
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.001, momentum=0.75, decay=1e-06, nesterov=True)
    adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])

    print "...Training starting..."

    training = model.fit(data,testLabels,batch_size=10,nb_epoch=20,verbose=1)

    print training.history
    print "...Training finished...."

    return model

def getPokerHand(model, array):
    t = model.predict(array, verbose=0)
    maxIndex = np.argmax(t)
    return maxIndex, t


#model = trainPokerNetwork()
#model.save("NeuralNetwork80inputs.h5")