"""
Sometimes, write deadly simple AI / Math codes and watch they run is all you need. #TooStressed

"""
import numpy as np
np.random.seed(233)

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras import optimizers

input_size = 5
output_size = 5

input = Input(shape=(input_size, ))
output = Dense(output_size, activation='sigmoid')(input)

model = Model(inputs=input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr = 0.1))

model.summary()

data = list()
labels = list()

for num in range(32):
    arr = np.array([])
    for i in range(5):
        k = (num >> i) & 1
        arr = np.append(arr, k)
    # print num, arr
    data.append(arr)
    labels.append(arr[::-1])
    
# print data
# print "Labels = "
# print labels
data = np.asarray(data)
labels = np.asarray(labels)

print "Data and labels are done"

model.fit(data, labels, epochs=50000)

print "All weights are:"
print model.get_layer(index=1).get_weights()

print "Model predictions are:"
out = model.predict(data)
print out
