import numpy as np


def load_and_split():

    X_train = np.load("../kmnist/kmnist-train-imgs.npz")['arr_0']
    y_train = np.load("../kmnist/kmnist-train-labels.npz")['arr_0']
    X_test = np.load("../kmnist/kmnist-test-imgs.npz")['arr_0']
    y_test = np.load("../kmnist/kmnist-test-labels.npz")['arr_0']

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    img_size = X_train.shape[1:]

    return X_train, X_test, y_train, y_test, train_size, test_size, img_size
