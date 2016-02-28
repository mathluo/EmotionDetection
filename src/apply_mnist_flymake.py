# demo for loading and applying a trained network. 

from __future__ import print_function
import sys , os, time, theano, pickle, lasagne, theano
import numpy as np
import theano.tensor as T
from load_data import *
from neural_network import *

# load the model from a .pkl file
foldername = '../TrainedModels/'
filename = 'train_mnist_best_model'
prediction_fn = load_and_build_model(foldername+filename)


# load the mnist data
X_test, y_test = load_mnist_test_data()


# predict( or do any stuff with the prediction)
acc = 0; 
test_num = 10000
start_time = time.time()
for i in range(test_num):
	cur_test = X_test[i,:,:,:]
	y_pred = prediction_fn(cur_test.reshape(1,1,28,28))
	if y_pred != y_test[i]:
		acc = acc+1

print ("test error \t\t{:.2f} %".format(float(acc)/float(test_num)*100))
print ("average function eval time \t\t{:.6f} ".format((time.time() - start_time)/test_num))
