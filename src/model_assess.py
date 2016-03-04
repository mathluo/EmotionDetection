# utiltity functions for testing
# Used for one time tests. For testing during training iterations, use theano compiled functions

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import pickle
import lasagne
from load_data import *



def compute_confusion_matrix(prediction_fn, X_test, y_test):
    number_of_classes = np.unique(y_test).shape[0]
    print (type(number_of_classes))
    confusion_matrix = np.zeros((number_of_classes,number_of_classes))
    y_predict = prediction_fn(X_test)
    y_test = y_test.astype(int)
    y_predict = y_predict.astype(int)
    for i in range(number_of_classes):
        number_class_i = (y_test == i).sum()
        for j in range(number_of_classes):
            confusion_matrix[i][j] =float( (y_predict[y_test == i] == j).sum())/float(number_class_i)
    return confusion_matrix


def compute_accuracy(prediction_fn, X_test,y_test):
    y_predict = prediction_fn(X_test)
    return float(np.equal(y_predict, y_test).sum())/float(y_test.shape[0])


    
def print_confusion_matrix(matrix,row_name,matrix_name):
    max_name_length = max(max([len(row_name[key]) for key in row_name])+2,9)
    print ('.'*(1+matrix.shape[0])*max_name_length)
    print ("Printing {}".format(matrix_name))
    print ("Row is Ground Truth")
    print ("Col is Prediction")
    print ('.'*(1+matrix.shape[0])*max_name_length)
    res = "".ljust(max_name_length)
    for foo, name  in row_name.iteritems():
        res  = res + name.ljust(max_name_length)
    print( res)
    for i in range(matrix.shape[0]):
        res = row_name[i].ljust(max_name_length)
        for j in range(matrix.shape[1]):
            res = res + "{:.2f}".format(matrix[i][j]*100).ljust(max_name_length)
        print (res )

    return
        
        
            





