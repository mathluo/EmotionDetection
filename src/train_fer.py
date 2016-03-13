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


batchsize = 300
num_epochs = 200
num_aug_dict = {'crop_flip':6, 'crop_rot_flip':12, 'rot_flip':4}
foldername = '../TrainedModelsBatch2/'

# Experiment 1
input_data = load_fer_dataset()
model_name = 'cnn'
train_model(input_data,model_name = model_name,foldername = foldername, name = 'fer1', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = None)
