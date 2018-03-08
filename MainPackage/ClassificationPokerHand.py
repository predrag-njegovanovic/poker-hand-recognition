from __future__ import division

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:, 0], ll[:,  1]] = 1
    return retVal


def trainPokerNetwork():
    f = open('Soft-dataset-pokerHand/PokerHand-Original.txt', 'r')
    textLines = []
    labels = []
    ranks = []
    numbers = []
    for line in f:
        newData = []
        ranks = []
        numbers = []
        parts = line.split(",")
        for i in range(0, len(parts), 2):
            if(i == 10):
                labels.append(int(parts[i]))
                break

            x = int(parts[i+1])
            ranks.append(int(parts[i]))
            numbers.append(x)

        ranks = sorted(ranks, key=int)
        numbers = sorted(numbers, key=int)
        highestNumber = numbers[-1]
        numbersExtracted = []
        numbersExtracted.append(highestNumber)
        ranksExtracted = []
        for s in range(len(ranks)-1):
            diff = ranks[s+1] - ranks[s]
            ranksExtracted.append(diff)

        for n in range(len(numbers)-1):
            diff = numbers[n+1] - numbers[n]
            numbersExtracted.append(diff)

        newData.extend(ranksExtracted)
        newData.extend(numbersExtracted)
        textLines.append(newData)

    print("Priprema ulaza.....")
    data = []
    for x in textLines:
        data.append(x)
    data = np.array(data)
    print("Zavrsena priprema.....")
    testLabels = np.array(labels)

    model = RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='entropy')
    model.fit(data, testLabels)
    return model


def getPokerHand(model, array):
    newData = []
    ranks = []
    numbers = []
    for i in range(0, len(array), 2):

        x = int(array[i])
        y = int(array[i+1])
        if (x == 15):
            x = 1
        elif (x == 16):
            x = 3
        elif (x == 17):
            x = 2
        elif (x == 18):
            x = 4

        if (y == 14):
            y = 13
        elif (y == 13):
            y = 12
        elif (y == 12):
            y = 11
        ranks.append(x)
        numbers.append(y)

    ranks = sorted(ranks, key=int)
    numbers = sorted(numbers, key=int)

    highestNumber = numbers[-1]
    numbersExtracted = []
    numbersExtracted.append(highestNumber)
    ranksExtracted = []
    for s in range(len(ranks) - 1):
        diff = ranks[s + 1] - ranks[s]
        ranksExtracted.append(diff)

    for n in range(len(numbers) - 1):
        diff = numbers[n + 1] - numbers[n]
        numbersExtracted.append(diff)

    newData.extend(ranksExtracted)
    newData.extend(numbersExtracted)

    data = np.array(newData)
    predict = model.predict(data)
    return predict
