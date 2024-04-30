import numpy as np

import wavelet_transform
from wavelet_transform import *


def load_data(filename, step, denoise):
    """
    Loads data from a .npy file and prepares it for training, validation, and testing.

    Parameters:
        filename (str): The path to the .npy file containing the dataset.
        step (int): The number of time steps to predict into the future.
        denoise (bool): Indicates whether to apply denoising.

    Returns:
        list: A list containing training, validation, and testing datasets,
              ground truth for the test set, and max/min values for denormalization.
    """

    # Load the dataset from the specified file
    data = np.load(filename)

    if denoise:
        cny_usd_prices = data[5]
        print(cny_usd_prices)
        denoised_prices = wavelet_denoise(cny_usd_prices)
        denoised_prices = denoised_prices.reshape(1, -1)
        data = np.vstack((data, denoised_prices))

    # Normalize the data to scale between -1 and 1
    max_data = np.max(data, axis=1, keepdims=True)  # Max value for each row
    min_data = np.min(data, axis=1, keepdims=True)  # Min value for each row
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)

    # Adjusted sizes for training and testing splits
    train_split = 4800  # The first 4800 data points for training
    test_split = data.shape[1] - train_split  # The rest for testing

    # Split the data into training, validation, and test sets
    x_train = data[:, :train_split]
    y_train = data[:, step:train_split + step]  # Target values offset by 'step'
    x_test = data[:, train_split:-step]
    y_test = data[:, train_split + step:]

    # Reshape data to add a singleton dimension, making it suitable for LSTM/SFM models
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    # Prepare the ground truth data for testing by selecting from 'step' index to the end
    gt_test = y_test

    # Return the prepared datasets and normalization parameters
    return [x_train, y_train, x_test, y_test, gt_test, max_data, min_data]
