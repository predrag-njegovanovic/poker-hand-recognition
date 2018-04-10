from __future__ import division
import os
import neural_network as nnwd

full_path = os.path.dirname(__file__)

if __name__ == "__main__":
    neural_network = nnwd.train_network()
    neural_network.save(full_path + '/models/neural_model.h5')
