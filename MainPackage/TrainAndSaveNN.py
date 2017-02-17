import NeuralNetwork as NN
import h5py
import NeuralNetworkWithDuplicates as nnwd

#model = NN.trainNetwork()
#model.save('CardNeural52.h5')

model = nnwd.trainNetwork()

model.save('CardNeuralDuplicates.h5')
