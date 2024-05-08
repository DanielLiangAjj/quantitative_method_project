import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np 
import h5py

import utils
from build import load_data
from sfm import SFM
from train import train_sfm,test_sfm
from utils import plot_results,ToVariable,use_cuda
from option_parser import OptionParser
from wavelet_transform import *

def main():
    parser = OptionParser()
    opt = parser.parse_args()
    # Specify the path to CSV file and the output NPY file path
    csv_filepath = 'Data/Foreign_Exchange_Rates.csv'
    npy_filepath = 'Data/output_data.npy'
    utils.convert_csv_to_npy(csv_filepath, npy_filepath)
    step = opt.step
    epochs = opt.epochs
    denoise = opt.denoise
    # splitting the dataset
    x_train, y_train, x_test, y_test, gt_test, max_data, min_data = load_data(npy_filepath,step,denoise)

    # Store the max and min values used for normalization in options for later use
    opt.max_data = max_data
    opt.min_data = min_data
    # Calculate lengths of training, validation, and test data
    opt.train_len = x_train.shape[1]
    # opt.val_len = x_val.shape[1]-x_train.shape[1]
    opt.test_len = x_test.shape[1]
    # Initialize the State Frequency Memory (SFM) model
    sfm = SFM(1,50,20,1)

    # Assign model and training/testing functions based on user option
    net = sfm
    train_function = train_sfm
    test_function = test_sfm

    print("Using net: %s"%(opt.net))

    # Train the model if training is enabled
    # if opt.train:
    train_function(net,x_train,y_train,epochs=epochs)
    # Test the model if testing is enabled
    # if opt.test:
    test_function(net,x_test,y_test,opt,denoise)

if __name__=='__main__':
    main()
