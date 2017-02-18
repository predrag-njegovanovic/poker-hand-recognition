from __future__ import division
import NeuralNetwork as NN
import h5py
import NeuralNetworkWithDuplicates as nnwd
import NeuralNetworkPokerHand as nnph
from keras.models import load_model
import numpy as np


#model = NN.trainNetwork()
#model.save('CardNeural52.h5')

#model = nnwd.trainNetwork()
#model = nnph.trainPokerNetwork()
#model.save('CardNeuralPoker.h5')
#


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), 'int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:, 0],ll[:,  1]] = 1
    return retVal




model = load_model('NeuralNetwork80inputs.h5')
arr = [15,1,15,13,17,4,17,3,15,12]
arr1 = [16,12,16,2,16,11,18,5,17,5]
arr2 = [16,10,17,10,15,2,17,11,18,9]

newData = []
for i in xrange(0, len(arr), 2):

    ranks = to_categorical([int(arr2[i])], 19)
    numbers = to_categorical([int(arr2[i + 1])], 14)

    r = ranks[0, 15:19]
    n = numbers[0, :]

    newData.extend(r)
    newData.extend(n)



data = np.array(newData)
data = np.resize(data, (1, 90))
num, predict = nnph.getPokerHand(model,data)
print num, predict