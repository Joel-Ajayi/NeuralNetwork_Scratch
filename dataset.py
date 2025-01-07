import numpy as np
from matplotlib import pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
import h5py
import tensorflow as tf

base_url = '/content/drive/My Drive/nn_tf'

class DataLoader:
    def __init__(self):
        pass

    def load_mnist_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        shape = x_train.shape[1:]
        x_train = x_train.reshape(x_train.shape[0], -1).T
        x_test = x_test.reshape(x_test.shape[0], -1).T
        # 0 - 10
        y_train = np.eye(10)[y_train].T
        y_test = np.eye(10)[y_test].T
        return (
            x_train[:, :1024] / 255.0,
            y_train[:, :1024],
            x_test / 255.0,
            y_test,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            shape,
        )

    def load_finger_signs_dataset(self):
        url = f"{base_url}/datasets/train_signs.h5"
        train_dataset = h5py.File(url, "r")
        test_dataset = h5py.File("datasets/test_signs.h5", "r")
        x_train, y_train = np.array(train_dataset["train_set_x"]), np.array(
            train_dataset["train_set_y"]
        )
        x_test, y_test = np.array(test_dataset["test_set_x"]), np.array(
            test_dataset["test_set_y"]
        )
        shape = x_train.shape[1:]
        x_train = x_train.reshape(x_train.shape[0], -1).T
        x_test = x_test.reshape(x_test.shape[0], -1).T
        # 0-5
        y_train = np.eye(6)[y_train].T
        y_test = np.eye(6)[y_test].T
        return (
            x_train / 255.0,
            y_train,
            x_test / 255.0,
            y_test,
            np.array([0, 1, 2, 3, 4, 5]),
            shape,
        )

    def load_cats_dataset(self):
        train_dataset = h5py.File(f"{base_url}/datasets/train_catvnoncat.h5", "r")
        train_set_x_orig = np.array(
            train_dataset["train_set_x"][:]
        )  # your train set features
        train_set_y_orig = np.array(
            train_dataset["train_set_y"][:]
        )  # your train set labels

        test_dataset = h5py.File(f"{base_url}/datasets/test_catvnoncat.h5", "r")
        test_set_x_orig = np.array(
            test_dataset["test_set_x"][:]
        )  # your test set features
        test_set_y_orig = np.array(
            test_dataset["test_set_y"][:]
        )  # your test set labels

        classes = np.array(test_dataset["list_classes"][:])  # the list of classes

        train_set_y = train_set_y_orig.reshape(train_set_y_orig.shape[0],1)
        test_set_y = test_set_y_orig.reshape(test_set_y_orig.shape[0], 1)

        train_set_x = train_set_x_orig.copy()
        test_set_x = test_set_x_orig.copy()

        return (
            tf.data.Dataset.from_tensor_slices((train_set_x,train_set_y)).shuffle(buffer_size=test_set_x.shape[0]),
            tf.data.Dataset.from_tensor_slices((test_set_x,test_set_y)),
            train_set_x_orig.shape[1:],
            classes,
        )

    def load_football_dataset(self):
        data = scipy.io.loadmat("datasets/data.mat")
        train_X = data["X"].T
        train_Y = data["y"].T
        test_X = data["Xval"].T
        test_Y = data["yval"].T

        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)

        return train_X, train_Y, test_X, test_Y
