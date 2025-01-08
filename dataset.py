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

    def load_finger_signs_dataset(self):
        url = f"{base_url}/datasets/train_signs.h5"
        train_dataset = h5py.File(url, "r")
        test_dataset = h5py.File(f"{base_url}/datasets/test_signs.h5", "r")
        x_train, y_train = np.array(train_dataset["train_set_x"]), np.array(
            train_dataset["train_set_y"]
        )
        x_test, y_test = np.array(test_dataset["test_set_x"]), np.array(
            test_dataset["test_set_y"]
        )
        shape = x_train.shape[1:]
        x_train = x_train.copy() / 255.0
        x_test = x_test.copy() / 255.0
        # 0-5
        y_train = np.eye(6)[y_train]
        y_test = np.eye(6)[y_test]
        return (
            tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=x_train.shape[0]),
            tf.data.Dataset.from_tensor_slices((x_test, y_test)),
            shape,
            np.array([0, 1, 2, 3, 4, 5]),
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
            tf.data.Dataset.from_tensor_slices((train_set_x,train_set_y)).shuffle(buffer_size=train_set_x.shape[0]),
            tf.data.Dataset.from_tensor_slices((test_set_x,test_set_y)),
            train_set_x_orig.shape[1:],
            classes,
        )
