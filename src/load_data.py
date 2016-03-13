#!/usr/bin/env python

"""
Load the MNIST Dataset
"""

from __future__ import print_function

import sys
import os
import time
from scipy.misc import imresize
import pandas as pd
import pickle
import numpy as np
import matplotlib.image as mpimg
from scipy.misc import imresize
import gzip


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.


def load_mnist_test_data():
    def load_mnist_images(foldername,filename):
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(foldername+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(foldername,filename):
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(foldername+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    foldername = '../Data/MNIST/'
    X_test = load_mnist_images(foldername,'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(foldername,'t10k-labels-idx1-ubyte.gz')
    return X_test, y_test #, X_test, y_test

def load_mnist_dataset():
    """ Load or Download MNIST dataset
    returns training,testing,validation data
    y_test containing numeric labels. 
    """
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(foldername,filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, foldername+filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.

    def load_mnist_images(foldername,filename):
        if not os.path.exists(foldername+filename):
            download(foldername,filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(foldername+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(foldername,filename):
        if not os.path.exists(foldername+filename):
            download(foldername,filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(foldername+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    foldername = '../Data/MNIST/'
    X_train = load_mnist_images(foldername,'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(foldername,'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(foldername,'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(foldername,'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val #, X_test, y_test




# ################## Load the SFEW dataset ##################
# Function for loading the SFEW dataset
#########################################################

def load_sfew_dataset():
    source = '../Data/SFEW_2/'  # I had to adjust this folder
    emotions = ['Angry', 'Disgust', 'Fear',
                'Happy', 'Sad', 'Surprise', 'Neutral']

    def load_face_images(folder_name):
        category = 0
        y_train = []
        list_of_images = []
        for emotion in emotions:
            category = category + 1
            data_folder = source + folder_name + '/' + \
                folder_name + '_Aligned_Faces/' + emotion + '/'
            for file in os.listdir(data_folder):
                if file.endswith(".png"):
                    img = mpimg.imread(data_folder + file)
                    img = imresize(img, (32, 32, 3))
                    img = img[:, :, :, np.newaxis]
                    img = img / 255.
                    list_of_images.append(img)
                    y_train.append(category)

        X_train = np.concatenate(tuple(list_of_images), axis=3)
        X_train = np.transpose(X_train, (3, 2, 0, 1))
        y_train = np.array(y_train, dtype=np.int32)

        return X_train, y_train
    X_train, y_train = load_face_images('Train')
    X_val, y_val = load_face_images('Val')
    # X_test, y_test = load_face_images('Test')
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_val.shape)
    # print(y_val.shape)

    # Just the training and validation dataset for now. 
    return X_train, y_train, X_val, y_val



# ################## Load the SFEW dataset ##################
# Function for loading the SFEW dataset
#########################################################

def load_fer_dataset():

    file_path = '../Data/fer2013/'
    if not (os.path.exists(file_path+'fpr_X_train.pkl') and os.path.exists(file_path+'fpr_y_train.pkl')\
            and os.path.exists(file_path+'fpr_X_val.pkl') and os.path.exists(file_path+'fpr_y_val.pkl')):
        file_name = 'fer2013.csv'
        data = pd.read_csv(file_path + file_name)
        num_rows = 48
        num_cols = 48
        # TODO: this lambda function is slow
        data['pixels'] = data['pixels'].apply(lambda x:
                                              np.reshape(np.fromstring(x, sep=' '),
                                                         (num_rows, num_cols)))

        X_train = data[data['Usage'] == 'Training']['pixels'].values
        X_train = [mtx[np.newaxis, np.newaxis, :, :] for mtx in X_train]
        X_train = np.concatenate(tuple(X_train), axis=0)
        X_train = X_train / 255.
        y_train = np.array(data[data['Usage'] == 'Training']['emotion'].values,
                           dtype=np.int32)

        X_val = data[data['Usage'] == 'PrivateTest']['pixels']
        X_val = [mtx[np.newaxis, np.newaxis, :, :] for mtx in X_val]
        X_val = np.concatenate(tuple(X_val), axis=0)
        X_val = X_val / 255.
        y_val = np.array(data[data['Usage'] == 'PrivateTest']['emotion'].values,
                         dtype=np.int32)
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_val.shape)
        # print(y_val.shape)

        output = open(file_path + 'fpr_X_train.pkl', 'wb')
        pickle.dump(X_train, output)
        output.close()

        output = open(file_path + 'fpr_y_train.pkl', 'wb')
        pickle.dump(y_train, output)
        output.close()

        output = open(file_path + 'fpr_X_val.pkl', 'wb')
        pickle.dump(X_val, output)
        output.close()

        output = open(file_path + 'fpr_y_val.pkl', 'wb')
        pickle.dump(y_val, output)
        output.close()
    else:
        X_train = pickle.load(open(file_path + 'fpr_X_train.pkl', "rb"))
        y_train = pickle.load(open(file_path + 'fpr_y_train.pkl', "rb"))
        X_val = pickle.load(open(file_path + 'fpr_X_val.pkl', "rb"))
        y_val = pickle.load(open(file_path + 'fpr_y_val.pkl', "rb"))

    return X_train, y_train, X_val, y_val



def drop_two_fer_classes():
    # drop the two classes: disgust and fear
    file_path = '../Data/fer2013/'
    X_train, y_train, X_val, y_val = load_fer_dataset()
    disgust_idx = np.where(y_train == 1)[0]
    fear_idx = np.where(y_train == 2)[0]
    rm_index = np.concatenate((disgust_idx, fear_idx))
    y_train_new = np.delete(y_train, rm_index, 0)
    X_train_new = np.delete(X_train, rm_index, 0)

    disgust_idx = np.where(y_val == 1)[0]
    fear_idx = np.where(y_val == 2)[0]
    rm_index = np.concatenate((disgust_idx, fear_idx))
    y_val_new = np.delete(y_val, rm_index, 0)
    X_val_new = np.delete(X_val, rm_index, 0)
    output = open(file_path + 'drop_2_fpr_X_train.pkl', 'wb')
    pickle.dump(X_train_new, output)
    output.close()

    output = open(file_path + 'drop_2_fpr_y_train.pkl', 'wb')
    pickle.dump(y_train_new, output)
    output.close()

    output = open(file_path + 'drop_2_fpr_X_val.pkl', 'wb')
    pickle.dump(X_val_new, output)
    output.close()

    output = open(file_path + 'drop_2_fpr_y_val.pkl', 'wb')
    pickle.dump(y_val_new, output)
    output.close()


def load_drop_2_class_dataset():
    file_path = '../Data/fer2013/'
    X_train = pickle.load(open(file_path + 'drop_2_fpr_X_train.pkl', "rb"))
    y_train = pickle.load(open(file_path + 'drop_2_fpr_y_train.pkl', "rb"))
    X_val = pickle.load(open(file_path + 'drop_2_fpr_X_val.pkl', "rb"))
    y_val = pickle.load(open(file_path + 'drop_2_fpr_y_val.pkl', "rb"))
    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    drop_two_fer_classes()
