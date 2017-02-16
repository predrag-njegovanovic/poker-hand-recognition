import NeuralNetwork as NN
import h5py


model = NN.trainNetwork()
model.save('CardNeural52.h5')