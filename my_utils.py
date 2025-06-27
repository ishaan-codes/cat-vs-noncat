import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    # dZ[Z <= 0] = 0 gives error in newer versions of numpy
    dZ = np.where(Z <= 0, 0, dZ)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

import h5py
def load_dataset():
    with h5py.File('train_catvnoncat.h5', "r") as train_file:
        train_x_orig = np.array(train_file["train_set_x"][:])
        train_y = np.array(train_file["train_set_y"][:])
    
    with h5py.File('test_catvnoncat.h5', "r") as test_file:
        test_x_orig = np.array(test_file["test_set_x"][:])
        test_y = np.array(test_file["test_set_y"][:])
    
    return train_x_orig, train_y, test_x_orig, test_y