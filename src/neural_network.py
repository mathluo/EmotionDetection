
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

def save_params(file_name, params):
    output = open(file_name + '.pkl', 'wb')
    pickle.dump(params, output)
    output.close()

def load_params(file_name):
    output = open(file_name + '.pkl', 'rb')
    result = pickle.load( output)
    output.close()   
    return result
 



# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None,input_shape = None, output_number = None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=input_shape,
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=output_number,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out




def build_cnn(input_var=None, input_shape = None,output_number = None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=output_number,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network



def build_bengio_net_1(input_var = None,input_shape = None,output_number = None): 
    # ConvNet1 implement structure by Bengio et al. 

     # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Stage 1:
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),pad = 'same')
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.LocalResponseNormalization2DLayer(network)

    # Stage 2:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),pad = 'same')
    network = lasagne.layers.Pool2DLayer(network, pool_size=(2, 2),pad = 'same',mode = 'average_exc_pad')
    network = lasagne.layers.LocalResponseNormalization2DLayer(network)

    # Stage 3: 
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),pad = 'same')
    network = lasagne.layers.Pool2DLayer(network, pool_size=(2, 2),pad = 'same',mode = 'average_exc_pad')
    # Final layers: 
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units = output_number,
            nonlinearity=lasagne.nonlinearities.softmax)    

    return network   



######################################## End of Model Building #####################################################3


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def choose_network(model_name,input_var,input_shape,output_number):
    if model_name == 'mlp':
        network = build_mlp(input_var, input_shape = input_shape, output_number = output_number)
    elif model_name == 'cnn':
        network = build_cnn(input_var, input_shape = input_shape, output_number = output_number)
    else:
        print("Unrecognized model type %r." % model)
        return
    return network

def get_input_shape(X):
    result = (None,)
    result = result + (X.shape[1],X.shape[2],X.shape[3])
    return result

def get_output_number(y):
    results = np.unique(y).shape[0]
    return results


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.






def train_model(input_data,model_name = 'cnn',name = 'train', batchsize = 100, num_epochs=100,**kwargs):
    # Set default options
    record_best_model = True
    if 'record_best_model' in kwargs:
        if  not kwargs['record_best_model']:
            record_best_model = False
    record_training_err = True
    if 'record_best_model' in kwargs:
        if  not kwargs['record_best_model']:
            record_best_model = False
    save_model = True
    if 'save_model' in kwargs:
        if not kwargs['save_model']:
            save_model = False


    #load the data
    print("Loading data...")
    X_train, y_train, X_val, y_val  = input_data
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')


    print("Building model and compiling functions...")
    input_shape = get_input_shape(X_train)
    output_number = get_output_number(y_train)
    network = choose_network(model_name, input_var, input_shape,output_number)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    learning_rate = 0.01
    momentum = 0.9
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=momentum)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],allow_input_downcast=True)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    if record_best_model:
        best_model_params = []
    if record_training_err:
        training_loss_record = np.zeros(num_epochs)
        validation_loss_record = np.zeros(num_epochs)
        validation_accuracy_record = np.zeros(num_epochs)

    min_val_err = np.inf
    min_val_accuracy = np.inf

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # record stuff down   
        train_err = train_err/train_batches
        val_err = val_err/val_batches
        val_acc = val_acc / val_batches 
        if record_training_err:
            training_loss_record[epoch] = train_err
            validation_loss_record[epoch] = val_err
            validation_accuracy_record[epoch] =  val_acc
        if val_err < min_val_err:
            min_val_err = val_err
            min_val_err_epoch = epoch
            min_val_accuracy = val_acc
            if record_best_model:
                best_model_params = lasagne.layers.get_all_param_values(network)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err ))
        print("  validation loss:\t\t{:.6f}".format(val_err ))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))

    print("  Best validation loss on iteration:\t\t{}".format(min_val_err_epoch))
    print("  Validation loss:\t\t{:.6f}".format(min_val_err))
    print("  Valication accuracy:\t\t{:.2f} %".format(min_val_accuracy*100))
    # After training, decide what to save and what to return 
    foldername = '../TrainedModels/'
    returns = (min_val_err,min_val_accuracy)
    if save_model:
        meta_data = {}
        meta_data['experiment_name'] = name
        meta_data['model_name'] = model_name
        meta_data['batchsize'] = batchsize
        meta_data['num_epochs'] = num_epochs
        meta_data['learning_rate'] = learning_rate
        meta_data['momentum'] = momentum
        save_params(foldername+name + '_experiment_metadata',meta_data)
        returns = returns+(min_val_err_epoch, )
        if record_training_err:
            returns = returns + (training_loss_record,validation_loss_record,validation_accuracy_record)
        save_params(foldername+name + '_loss_record',returns)
        if record_best_model:
            best_model_params = (model_name, input_shape, output_number, best_model_params)
            save_params(foldername+name + '_best_model', best_model_params)
            returns = returns+(best_model_params,)
    return returns


def load_and_build_model(filename):
    # load the model from a .pkl file
    model_name, input_shape,output_number, model_params = load_params(filename)
    input_var = T.tensor4('inputs')
    network = choose_network(model_name, input_var, input_shape, output_number)
    lasagne.layers.set_all_param_values(network,model_params)

    #compile the final output function
    prediction = T.argmax(lasagne.layers.get_output(network, deterministic=True),axis = 1)
    prediction_fn = theano.function( [input_var], prediction, allow_input_downcast = True)
    return prediction_fn







# Net from Kaggle. 
# 48x48x1fm
# Mirror image
# Rotation and scale (center +-3 in both directions, angle +-45 degree, scale 0.8-1.2), cropping result to 42x42x1fm
# Convolutional layer (fully connected wrt feature maps) with 5x5 weighting windows, 42x42x32fm, activation function - rectified linear
# Max subsampling layer, 21x21x32fm
# Convolutional layer (fully connected wrt feature maps) with 4x4 weighting windows, 20x20x32fm, activation function - rectified linear
# Average subsampling layer, 10x10x32fm
# Convolutional layer (fully connected wrt feature maps) with 5x5 weighting windows, 10x10x64fm, activation function - rectified linear
# Average subsampling layer, 5x5x64fm
# Fully connected with 20% dropout of output neurons, 3072 output neurons, activation function - rectified linear 
    





