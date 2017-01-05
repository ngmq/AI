# Simple 2D CNN on MINST dataset by Keras

from __future__ import print_function
import numpy as np
np.random.seed(233)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
