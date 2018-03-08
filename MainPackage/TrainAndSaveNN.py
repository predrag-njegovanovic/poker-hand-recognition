from __future__ import division
import NeuralNetworkWithDuplicates as nnwd


if __name__ == "__main__":
    modelNeural = nnwd.trainNetwork()
    modelNeural.save('CardNeuralPoker.h5')
