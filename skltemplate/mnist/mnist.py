# -*- coding: utf-8 -*-
import numpy as np
import struct
import os


def load_mnist():
    print("loading Data")
    dirname = os.path.dirname(__file__)
    #filename = os.path.join(dirname, 'src/MNIST/t10k-images-idx3-ubyte')
    
    """Load MNIST data"""
    with open(os.path.join(dirname, 'train-labels-idx1-ubyte'), 'rb') as f:
        struct.unpack(">II", f.read(8))
        y_train = np.fromfile(f, dtype=np.int8)

    with open(os.path.join(dirname, 'train-images-idx3-ubyte'), 'rb') as f:
        _, _, rows, cols = struct.unpack(">IIII", f.read(16))
        X_train = np.fromfile(f, dtype=np.uint8).reshape(len(y_train), rows, cols)

    with open(os.path.join(dirname, 't10k-labels-idx1-ubyte'), 'rb') as f:
        struct.unpack(">II", f.read(8))
        y_test = np.fromfile(f, dtype=np.int8)

    with open(os.path.join(dirname, 't10k-images-idx3-ubyte'), 'rb') as f:
        _, _, rows, cols = struct.unpack(">IIII", f.read(16))
        X_test = np.fromfile(f, dtype=np.uint8).reshape(len(y_test), rows, cols)

    # flatten
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    # convert to one-hot encoding
    one_hot = np.zeros((len(y_train), 10))
    one_hot[np.arange(len(y_train)), y_train] = 1
    y_train = one_hot

    one_hot = np.zeros((len(y_test), 10))
    one_hot[np.arange(len(y_test)), y_test] = 1
    y_test = one_hot

    return X_train, y_train, X_test, y_test