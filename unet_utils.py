import numpy as np
import h5py


def load_dataset():
    dataset = h5py.File(DATA_PATH, "r")

    train_set_x_orig = np.array(dataset["train_img"][:])
    train_set_y_orig = np.array(dataset["train_label"][:])

    test_set_x_orig = np.array(dataset["test_img"][:])
    sizes_test = np.array(dataset["sizes_test"][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, sizes_test
