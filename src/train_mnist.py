from __future__ import print_function

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import pickle
import lasagne
import load_data
import neural_network
reload(load_data)
reload(neural_network)
from load_data import *
from neural_network import *


# load the data
input_data = load_mnist_dataset()
# choose the network
model_name = 'cnn'
#start training
train_model(input_data,model_name,name = 'mnist1', batchsize = 300, num_epochs=200)





