# script for preprocessing:(i.e. augmentation, normalization) of FER dataset

from __future__ import print_function

import sys
import os
import time
import scipy as sp
import pandas as pd
import pickle
import numpy as np
import matplotlib.image as mpimg
from scipy.misc import imresize
import gzip
import load_data
reload(load_data)
from load_data import *


def normalize_batch(img_batch):
# assume input is a [n,1,48,48] tensor
    img_batch = img_batch.astype(float)
    results = img_batch - np.mean(img_batch,axis = (2,3)).reshape(img_batch.shape[0],1,1,1) 
    results = results /  np.std(results,axis = (2,3)).reshape(img_batch.shape[0],1,1,1) 
    return results

def normalize_single_image(img):
    img = img.astype(float)
    results = img - np.mean(img)
    results = results / np.std(results)
    return results


def random_crop(img,crop_sz,left_shift, right_shift):
    default_shift = int((img.shape[0] - crop_sz)/2)
    left_shift = left_shift + default_shift
    right_shift = right_shift + default_shift
    return img[left_shift:left_shift + crop_sz,right_shift:right_shift + crop_sz]
    


def aug_single_crop_rot_flip(img,crop_sz = 42): 
    # generate random shifts for crop
    imsz = crop_sz
    if img is None:
        return (12,imsz) # the expansion number
    margin = int((img.shape[0] - crop_sz)/2)
    left_shift = np.random.randint(low = -margin, high = margin+1, size = 12)
    right_shift = np.random.randint(low = -margin, high = margin+1, size = 12)
    # mirror
    mir_img = np.fliplr(img)
    # rotate
    start_idx = np.floor(0.1 * img.shape[0])
    end_idx = np.floor(0.9 * img.shape[0])
    degree = 15
    rot_img_1 = sp.misc.imrotate(img, degree)
    rot_img_1 = rot_img_1[start_idx:end_idx, start_idx:end_idx]
    rot_img_1 = sp.misc.imresize(rot_img_1, (img.shape[0], img.shape[1]))
    rot_img_1 = rot_img_1.astype(float)/255
    rot_img_2 = sp.misc.imrotate(img, -degree)
    rot_img_2 = rot_img_2[start_idx:end_idx, start_idx:end_idx]
    rot_img_2 = sp.misc.imresize(rot_img_2, (img.shape[0], img.shape[1]))
    rot_img_2 = rot_img_2.astype(float)/255
    # save results to temp
    temp = np.zeros((4, img.shape[0], img.shape[1]))
    temp[0, :, :] = img
    temp[1, :, :] = mir_img
    temp[2, :, :] = rot_img_1 
    temp[3, :, :] = rot_img_2 
    # three random crops to all of temp
    res = np.zeros((12, 1, imsz, imsz))
    for i in range(4):
        for j in range(3):
            res[i*3+j,0,:,:] = random_crop(temp[i,:,:], crop_sz, left_shift[i*3+j],right_shift[i*3+j])
    return res

def aug_single_crop_flip(img, crop_sz = 42): 
    # generate random shifts for crop
    imsz = crop_sz
    if img is None:
        return (6,imsz) # the expansion number
    margin = int((img.shape[0] - crop_sz)/2)
    left_shift = np.random.randint(low = -margin, high = margin+1, size = 12)
    right_shift = np.random.randint(low = -margin, high = margin+1, size = 12)
    # mirror
    mir_img = np.fliplr(img)
    # # rotate
    # start_idx = np.floor(0.1 * img.shape[0])
    # end_idx = np.floor(0.9 * img.shape[0])
    # degree = 15
    # rot_img_1 = sp.misc.imrotate(img, degree)
    # rot_img_1 = rot_img_1[start_idx:end_idx, start_idx:end_idx]
    # rot_img_1 = sp.misc.imresize(rot_img_1, (img.shape[0], img.shape[1]))
    # rot_img_1 = rot_img_1.astype(float)/255
    # rot_img_2 = sp.misc.imrotate(img, -degree)
    # rot_img_2 = rot_img_2[start_idx:end_idx, start_idx:end_idx]
    # rot_img_2 = sp.misc.imresize(rot_img_2, (img.shape[0], img.shape[1]))
    # rot_img_2 = rot_img_2.astype(float)/255
    # save results to temp
    temp = np.zeros((2, img.shape[0], img.shape[1]))
    temp[0, :, :] = img
    temp[1, :, :] = mir_img
    # temp[2, :, :] = rot_img_1 
    # temp[3, :, :] = rot_img_2 
    # three random crops to all of temp
    res = np.zeros((6, 1, imsz, imsz))
    for i in range(2):
        for j in range(3):
            res[i*3+j,0,:,:] = random_crop(temp[i,:,:], crop_sz, left_shift[i*3+j],right_shift[i*3+j])
    return res


def aug_single_rot_flip(img): 
    if img is None:
        return (4,48) # hard coded for now
    start_idx = np.floor(0.1 * img.shape[0])
    end_idx = np.floor(0.9 * img.shape[0])
    # mirror
    mir_img = np.fliplr(img)
    # rotate
    degree = 15
    rot_img_1 = sp.misc.imrotate(img, degree)
    rot_img_1 = rot_img_1[start_idx:end_idx, start_idx:end_idx]
    rot_img_1 = sp.misc.imresize(rot_img_1, (img.shape[0], img.shape[1]))
    rot_img_1 = rot_img_1.astype(float)/255
    rot_img_2 = sp.misc.imrotate(img, -degree)
    rot_img_2 = rot_img_2[start_idx:end_idx, start_idx:end_idx]
    rot_img_2 = sp.misc.imresize(rot_img_2, (img.shape[0], img.shape[1]))
    rot_img_2 = rot_img_2.astype(float)/255
    res = np.zeros((4, 1, img.shape[0], img.shape[1]))
    # three random crops
    res[0, 0, :, :] = img
    res[1, 0, :, :] = mir_img
    res[2, 0, :, :] = rot_img_1
    res[3, 0, :, :] = rot_img_2
    return res


def aug_batch_img(batch_x, batch_y,shuffle = True, mode = 'rot_flip'):
    if mode == 'rot_flip':
        aug_func = aug_single_rot_flip
    elif mode == 'crop_flip':
        aug_func = aug_single_crop_flip
    elif mode == 'crop_rot_flip':
        aug_func = aug_single_crop_rot_flip
    num_copy, imsz = aug_func(img = None)
    res_1= np.zeros((batch_x.shape[0]*num_copy,batch_x.shape[1], imsz, imsz))
    res_2 = np.zeros(batch_y.shape[0]*num_copy).astype(int)
    for i in range(batch_x.shape[0]):
        res_1[i*num_copy:(i+1)*num_copy, :, :, :] = aug_func(batch_x[i,0,:, :])
        res_2[i*num_copy:(i+1)*num_copy]  = np.ones(num_copy).astype(int)*batch_y[i]
    if shuffle:
        indices = range(batch_x.shape[0]*num_copy)
        np.random.shuffle(indices)
        res_1 = res_1[indices,:,:,:]
        res_2 = res_2[indices]
    return res_1 ,res_2
 





def load_augment_data(mode = 'crop_rot_flip',drop_std_flag = False):
    file_path = '../Data/fer2013/'
    X_train = pickle.load( open( file_path + 'fpr_X_train.pkl', "rb" ) )
    y_train = pickle.load( open( file_path + 'fpr_y_train.pkl', "rb" ) )
    X_val = pickle.load( open( file_path + 'fpr_X_val.pkl', "rb" ) )
    y_val = pickle.load( open( file_path + 'fpr_y_val.pkl', "rb" ) )

    if drop_std_flag:
        X_train, y_train = drop_std(X_train,y_train,up = 300, low = 300)
        X_val, y_val = drop_std(X_val,y_val,up = 10, low = 10)
    print('loading original complete')
    X_train_aug, y_train_aug = aug_batch_img(X_train, y_train, mode = mode)
    print('augmented training data')
    if not(mode == 'rot_flip'):
        X_val_aug, y_val_aug = aug_batch_img(X_val, y_val, shuffle = False, mode = 'crop_flip')
    else:
        X_val_aug = X_val
        y_val_aug = y_val
    print('augmented validation data')
    if drop_std_flag:
        X_train_aug = normalize_batch(X_train_aug)
        X_val_aug = normalize_batch(X_train_aug)

    return X_train_aug, y_train_aug, X_val_aug, y_val_aug

# print ('saving data ......')
# output = open(file_path + 'fpr_X_train_aug_crf.pkl', 'wb')
# cPickle.dump(X_train_aug, output)
# output.close()

# output = open(file_path_aug + 'fpr_y_train_aug_crf.pkl', 'wb')
# cPickle.dump(y_train_aug, output)
# output.close()


# output = open(file_path + 'fpr_X_val_aug_crf.pkl', 'wb')
# cPickle.dump(X_val_aug, output)
# output.close()

# output = open(file_path + 'fpr_y_val_aug_crf.pkl', 'wb')
# cPickle.dump(y_val_aug, output)
# output.close()

