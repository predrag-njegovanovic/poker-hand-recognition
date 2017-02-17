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
model = load_model('CardNeuralPoker.h5')
arr = [15,1,15,13,17,4,17,3,15,12]
arr1 = [16,12,16,2,16,11,18,5,17,5]
arr2 = [16,10,17,7,15,2,17,11,18,9]

newData = []
for i in xrange(0, len(arr), 2):
    rank = (int(arr[i]) - 15) / 3  # (x- minValue)/(maxValue - minValue) #normalization 0 - 1
    number = (int(arr[i + 1]) - 1) / 13
    newData.append(rank)
    newData.append(number)

data = np.array(newData)
data = np.resize(data, (1, 10))
num, predict = nnph.getPokerHand(model,data)
print num, predict