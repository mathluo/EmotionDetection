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
num_epochs = 100
num_aug_dict = {'crop_flip':6, 'crop_rot_flip':12, 'rot_flip':4}
foldername = '../TrainedModelsBatch3/'

# # Experiment 1
# X_train,y_train,X_val,y_val = load_drop_2_class_dataset()
# #X_train = normalize_batch(X_train)
# #X_val = normalize_batch(X_val)
# input_data = (X_train,y_train,X_val,y_val)
# model_name = 'cnn'
# train_model(input_data,model_name = model_name,foldername = foldername, name = 'fer1', batchsize = batchsize, 
# 	num_epochs=num_epochs, num_aug = None)#num_aug = num_aug_dict['crop_flip'])


# # Experiment 2
# X_train,y_train,X_val,y_val = five_class_drop_300()
# X_train = normalize_batch(X_train)
# X_val = normalize_batch(X_val)
# input_data = (X_train,y_train,X_val,y_val)
# model_name = 'cnn'
# train_model(input_data,model_name = model_name,foldername = foldername, name = 'fer1_normalize', batchsize = batchsize, 
# 	num_epochs=num_epochs, num_aug = None)#num_aug = num_aug_dict['crop_flip'])


# Experiment 3
X_train,y_train,X_val,y_val = five_class_drop_300()
X_train,y_train = aug_batch_img(X_train,y_train,shuffle = True, mode = 'rot_flip')
X_val,y_val = aug_batch_img(X_val,y_val,shuffle = True, mode = 'rot_flip')
X_train = normalize_batch(X_train)
X_val = normalize_batch(X_val)
input_data = (X_train,y_train,X_val,y_val)
model_name = 'cnn'
train_model(input_data,model_name = model_name,foldername = foldername, name = 'fer1_aug_rf_normalize', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = None)#num_aug = num_aug_dict['crop_flip'])


# Experiment 4
model_name = 'cnn2'
train_model(input_data,model_name = model_name,foldername = foldername, name = 'fer2_aug_rf_normalize', batchsize = batchsize, 
	num_epochs=num_epochs, num_aug = None)#num_aug = num_aug_dict['crop_flip'])