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
from data_preprocess import *



batchsize = 300
num_epochs = 300
num_aug_dict = {'crop_flip':6, 'crop_rot_flip':12, 'rot_flip':4}
foldername = '../TrainedModelsBatch1/'

# Experiment 1
input_data = load_fer_dataset()
model_name = 'cnn'
train_model(input_data,model_name = model_name,foldername = foldername, name = 'fer1', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = None)#num_aug = num_aug_dict['crop_flip'])


# Experiment 2
model_name = 'cnn2'
train_model(input_data,model_name = model_name,foldername = foldername,name = 'fer2', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = None)



# Experiment 3
input_data = load_augment_data(mode = 'crop_flip')
model_name = 'cnn'
train_model(input_data,model_name = model_name,foldername = foldername,name = 'fer1_aug_cf', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = num_aug_dict['crop_flip'])


# Experiment 4
# input_data = load_augment_data(mode = 'crop_flip')
model_name = 'cnn2'
train_model(input_data,model_name = model_name,foldername = foldername,name = 'fer2_aug_cf', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = num_aug_dict['crop_flip'])


# Experiment 5
input_data = load_augment_data(mode = 'rot_flip')
model_name = 'cnn'
train_model(input_data,model_name = model_name,foldername = foldername,name = 'fer1_aug_rf', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = None)


# Experiment 6
# input_data = load_augment_data(mode = 'rot_flip')
model_name = 'cnn2'
train_model(input_data,model_name = model_name,foldername = foldername,name = 'fer2_aug_rf', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = None)


# Experiment 7
input_data = load_augment_data(mode = 'crop_rot_flip')
model_name = 'cnn'
train_model(input_data,model_name = model_name,foldername = foldername,name = 'fer1_aug_crf', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = num_aug_dict['crop_flip'])


# Experiment 8
# input_data = load_augment_data(mode = 'crop_rot_flip')
model_name = 'cnn2'
train_model(input_data,model_name = model_name,foldername = foldername,name = 'fer2_aug_crf', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = num_aug_dict['crop_flip'])
