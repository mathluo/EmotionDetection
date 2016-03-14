# script for assessing the model FER1

from __future__ import print_function
import sys , os, time, theano, pickle, lasagne, theano
import numpy as np
import theano.tensor as T
from load_data import *
from neural_network import *
from model_assess import *

# load the model from a .pkl file
foldername = '../TrainedModelsBatch3/'
filename = 'fer1_normalize_best_model'
prediction_fn = load_and_build_model(foldername+filename)


# load the mnist data
X_train, y_train, X_val, y_val = five_class_drop_300()

row_name = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

row_name_5 = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Neutral'}                
test_name = "FER1"

# predict( or do any stuff with the prediction)
print ("validation accuracy {:.2f}".format(100*compute_accuracy(prediction_fn, X_val, y_val)))
print_confusion_matrix(compute_confusion_matrix(prediction_fn,X_val,y_val),row_name_5,test_name)
