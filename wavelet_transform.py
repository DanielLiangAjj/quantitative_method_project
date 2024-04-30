#Importing Linraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import keras
import math
import yfinance as yf
import numpy as np
import pandas as pd
import pywt
from pywt import wavedec
from pywt import waverec
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def wavelet_denoise(data, level=2, wavelet='db2'):
    """ Apply wavelet denoising to a signal. """

    def mad(arr):
        """ Median Absolute Deviation: a more robust estimator for the std deviation """
        return np.median(np.abs(arr - np.median(arr)))

    # Decomposition
    coeff = wavedec(data, wavelet, level=level)

    # Thresholding
    sigma = mad(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]

    # Reconstruction
    denoised = waverec(coeff, wavelet)
    return denoised[:len(data)]




