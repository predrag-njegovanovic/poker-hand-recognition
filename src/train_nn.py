from __future__ import division
import neural_network as nnwd


if __name__ == "__main__":
    modelNeural = nnwd.trainNetwork()
    modelNeural.save('CardNeuralPoker.h5')
