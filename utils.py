import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

use_cuda = torch.cuda.is_available()

def recover(data,length=50):
    for i in range(1,length):
        data[i,:,:] = data[0,:,:]
    return data

def plot_results(predicted_data, true_data):
    # use in train.py 
    # plot evaluate result
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def ToVariable(x):
    # use in train.py 
    # change from numpy.array to torch.variable   
    tmp = torch.FloatTensor(x)
    if use_cuda:
        return Variable(tmp).cuda()
    else:
        return Variable(tmp)

def shuffle_data(X,Y):
    data_train = [(x,y) for x,y in zip(X, Y)]
    np.random.shuffle(data_train)
    X = np.array([x for x,y in data_train])
    Y = np.array([y for x,y in data_train])
    return X,Y



def convert_csv_to_npy(csv_filepath, npy_filepath):
    # Load the CSV file
    data = pd.read_csv(csv_filepath, na_values=['ND'], index_col=0)
    data.fillna(method='ffill', inplace=True)
    # If there are still NaN values from the beginning of the dataset, back-fill those
    data.fillna(method='bfill', inplace=True)
    data.drop(data.columns[0], axis=1, inplace=True)
    # print(data.head())
    # print(data)
    # Convert the DataFrame to a NumPy array
    # data = data.drop('Time Serie', axis=1)
    data_array = data.values.T

    # Save the array to an NPY file
    np.save(npy_filepath, data_array)

    print(f"Data saved to {npy_filepath}")





