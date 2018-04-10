from __future__ import division

import os
import numpy as np
import pickle as pick

from sklearn.ensemble import RandomForestClassifier

full_path = os.path.dirname(__file__)


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:, 0], ll[:,  1]] = 1
    return retVal


def train_random_forest():
    f = open(full_path + '/data/poker_hand_dataset/PokerHand-Original.txt', 'r')
    text_lines = []
    labels = []
    ranks = []
    numbers = []
    for line in f:
        new_data = []
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
        highest_number = numbers[-1]
        numbers_extracted = []
        numbers_extracted.append(highest_number)
        ranks_extracted = []
        for s in range(len(ranks)-1):
            diff = ranks[s+1] - ranks[s]
            ranks_extracted.append(diff)

        for n in range(len(numbers)-1):
            diff = numbers[n+1] - numbers[n]
            numbers_extracted.append(diff)

        new_data.extend(ranks_extracted)
        new_data.extend(numbers_extracted)
        text_lines.append(new_data)

    print("Priprema ulaza.....")
    data = []
    for x in text_lines:
        data.append(x)
    data = np.array(data)
    print("Zavrsena priprema.....")
    testLabels = np.array(labels)

    model = RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='entropy')
    model.fit(data, testLabels)
    with open(full_path + '/models/random_forest.pkl', 'wb') as f:
        pick.dump(model, f)
    return model


def get_poker_hand(model, array):
    new_data = []
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

    highest_number = numbers[-1]
    numbers_extracted = []
    numbers_extracted.append(highest_number)
    ranks_extracted = []
    for s in range(len(ranks) - 1):
        diff = ranks[s + 1] - ranks[s]
        ranks_extracted.append(diff)

    for n in range(len(numbers) - 1):
        diff = numbers[n + 1] - numbers[n]
        numbers_extracted.append(diff)

    new_data.extend(ranks_extracted)
    new_data.extend(numbers_extracted)

    data = np.array(new_data)
    data = data.reshape(1, -1)
    predict = model.predict(data)
    return predict


if __name__ == '__main__':
    train_random_forest()
