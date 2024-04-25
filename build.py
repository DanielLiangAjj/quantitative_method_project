import numpy as np


def load_data(filename, step):
    """
    Loads data from a .npy file and prepares it for training, validation, and testing.

    Parameters:
        filename (str): The path to the .npy file containing the dataset.
        step (int): The number of time steps to predict into the future.

    Returns:
        list: A list containing training, validation, and testing datasets,
              ground truth for the test set, and max/min values for denormalization.
    """

    # Load the dataset from the specified file
    data = np.load(filename)

    # Prepare the ground truth data for testing by selecting from 'step' index to the end
    gt_test = data[:, step:]

    # Normalize the data to scale between -1 and 1
    max_data = np.max(data, axis=1, keepdims=True)  # Max value for each row
    min_data = np.min(data, axis=1, keepdims=True)  # Min value for each row
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)

    # Determine indices for splitting the dataset into training, validation, and testing
    train_split = int(0.8 * data.shape[1])  # 80% for training
    val_split = int(0.9 * data.shape[1])  # Next 10% for validation

    # Split the data into training, validation, and test sets
    x_train = data[:, :train_split]
    y_train = data[:, step:train_split + step]  # Target values offset by 'step'
    x_val = data[:, :val_split]
    y_val = data[:, step:val_split + step]  # Target values offset by 'step'
    x_test = data[:, :-step]
    y_test = data[:, step:]

    # Reshape data to add a singleton dimension, making it suitable for LSTM/SFM models
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    # Return the prepared datasets and normalization parameters
    return [x_train, y_train, x_val, y_val, x_test, y_test, gt_test, max_data, min_data]
