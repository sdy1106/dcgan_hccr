import sys
sys.path.append('..')

import numpy as np
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt

from lib.data_utils import shuffle
from lib.config import data_dir

# delete all-zeros-colunms
def delete_all_zero_columns(C_):
    C = np.zeros((100, 1000, 123))
    cnt = 0
    for i in range(123):
        if np.max(C_[:, :, i]) != 0:
            C[:, :, cnt] = C_[:, :, i]
            cnt += 1
    C = C[:, :, 0:cnt]
    return C

def mnist():
    fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))
    
    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY

def mnist_with_valid_set():
    trX, teX, trY, teY = mnist()

    trX, trY = shuffle(trX, trY)
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY

def load(data_dir , image_file_name , label_file_name , code_file_name ,train_val_test_rate = [0.8 , 0.1 , 0.1]):

    X = np.load(os.path.join(data_dir , image_file_name)) #100*1000*28*28
    Y = np.load(os.path.join(data_dir , label_file_name)) #100*1000
    C = delete_all_zero_columns(np.load(os.path.join(data_dir , code_file_name))) #100*1000*code_len

    assert np.shape(X)[1] == np.shape(Y)[1] == np.shape(C)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_val_test_rate[0])
    val_num = int(np.shape(X)[1] * train_val_test_rate[1])
    test_num = int(np.shape(X)[1] * train_val_test_rate[2])

    train_X = X[:,0:train_num,:]
    train_Y = C[:,0:train_num,:]

    val_X = X[: , train_num:train_num+val_num , :]
    val_Y = C[: , train_num:train_num+val_num , :]

    test_X = X[:,train_num+val_num: , :]
    test_Y = C[:,train_num+val_num: , :]

    code_len = np.shape(C)[2]

    return train_X ,  val_X ,  test_X , train_Y , val_Y , test_Y , code_len
    # train_X[100*800*28*28] train_Y[100*800*code_length]
