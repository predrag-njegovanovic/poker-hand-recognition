from __future__ import division
import numpy as np
from sklearn.ensemble import RandomForestClassifier


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
        ranks = []
        numbers = []
        parts = line.split(",")
        for i in xrange(0, len(parts), 2):
            if(i == 10):
                 labels.append(int(parts[i]))
                 break

            x = int(parts[i])
            if (x == 15):
                r = 1
            elif (x == 16):
                r = 2
            elif (x == 17):
                r = 3
            elif (x == 18):
                r = 4
            ranks.append(r)
            numbers.append(int(parts[i+1]))

        ranks = sorted(ranks, key=int)
        numbers = sorted(numbers, key=int)
        newData.extend(numbers)
        newData.extend(ranks)
        textLines.append(newData)


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
    testLabels = np.array(labels)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(data, testLabels)

    return model

def getPokerHand(model, array):
    newData = []
    ranks = []
    numbers = []
    for i in xrange(0, len(array), 2):

        x = int(array[i])
        if (x == 15):
            r = 1
        elif (x == 16):
            r = 2
        elif (x == 17):
            r = 3
        elif (x == 18):
            r = 4
        ranks.append(r)
        numbers.append(int(array[i + 1]))

    ranks = sorted(ranks, key=int)
    numbers = sorted(numbers, key=int)

    newData.extend(numbers)
    newData.extend(ranks)

    data = np.array(newData)
    predict = model.predict(data)
    return predict

