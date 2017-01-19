from keras.models import Sequential
from keras.layers import Embedding, Convolution1D, Convolution2D, Dense, Activation
import numpy as np

np.random.seed(233)

model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(low = -10, high = 10, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
# assert output_array.shape == (32, 10, 64)
print output_array