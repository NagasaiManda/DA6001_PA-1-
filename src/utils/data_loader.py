"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist

def load_data(dataset):
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28*28).astype("float32")
    x_test = x_test.reshape(-1, 28*28).astype("float32")
    x_train /= 255
    x_test /= 255
    y_train_out = np.zeros((y_train.shape[0], 10))
    y_train_out[np.arange(y_train.shape[0]), y_train] = 1
    y_test_out = np.zeros((y_test.shape[0], 10))
    y_test_out[np.arange(y_test.shape[0]), y_test] = 1
    return (x_train, y_train_out), (x_test, y_test_out)


    