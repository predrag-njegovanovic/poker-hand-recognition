from __future__ import division
import h5py
import NeuralNetworkWithDuplicates as nnwd
import ClassificationPokerHand as nnph
import pickle as pick


modelNeural = nnwd.trainNetwork()
#modelRandomForest = nnph.trainPokerNetwork()
modelNeural.save('CardNeuralPoker.h5')
#with open("RandomForestPokerHand.bin", 'wb') as fp:
#   pick.dump(modelRandomForest,fp,pick.HIGHEST_PROTOCOL)
