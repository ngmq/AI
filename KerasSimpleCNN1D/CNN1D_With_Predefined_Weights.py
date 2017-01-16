from keras.models import Sequential
from keras.layers import Convolution1D, Convolution2D, Dense, Activation
import numpy as np

np.random.seed(233)

## Conventional Feedforward NN
# model = Sequential()
# model.add(Dense(output_dim = 3, input_dim = 2))
# model.add(Dense(output_dim = 3))
# print model.summary()
# print model.output_shape # return (None, 3) = number of neurons in the last layer

## Convolutional NN

## Input shape: 3x2 == 3 words, 2 features each
## Filter length = 1, each filter has 2 weights connect to 2 features of each word

## One filter 
model1 = Sequential()
_W = np.array([[[[0.1], [0.2]]]]) ## weight from filter to 1st feature = 0.1, to 2nd feature = 0.2
_B = np.array([0.5])
ww = [_W, _B]
model1.add(Convolution1D(nb_filter = 1, filter_length = 1, border_mode = 'same', input_shape = (3, 2), weights = ww))
print model1.summary()
print 'Model.input_shape = ', model1.input_shape
print 'Model.output_shape = ', model1.output_shape
x = np.array([[[1, 1], [2, 2], [3, 3]]])
# print(x.shape)
# print(x.shape[0])
print(len(x.shape))
y = model1.predict(x)
print y
## y = [[[0.8000] = 1x0.1 + 1x0.2 + 0.5
##		 [1.1000] = 2x0.1 + 2x0.2 + 0.5
##		 [1.4000] = 3x0.1 + 3.0.2 + 0.5
###    ]]]
print y.shape

## Two filter
model2 = Sequential()
_W = np.array([[[[0.1, 0.2], [0.2, 0.4]]]])
_B = np.array([0.5, 1])
ww = [_W, _B]
model2.add(Convolution1D(nb_filter = 2, filter_length = 1, border_mode = 'same', input_shape = (3, 2), weights = ww))
print model2.summary()
print 'Model.input_shape = ', model2.input_shape
print 'Model.output_shape = ', model2.output_shape
x = np.array([[[1, 1], [2, 2], [3, 3]]])
# print(x.shape)
# print(x.shape[0])
print(len(x.shape))
y = model2.predict(x)
print y
## y = [[[ 0.80000001  1.60000002]
##       [ 1.10000002  2.20000005]
##       [ 1.4000001   2.80000019]]]
print y.shape
