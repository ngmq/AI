import numpy as np

np.random.seed(7)

import pickle
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
from sklearn.preprocessing import normalize
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K
import matplotlib.pyplot as plt

if K.image_data_format() == "channels_last":
    input_shape = (128, 128, 3)
else:
    input_shape = (3, 128, 128)
    
model = Sequential()

# Block 1
model.add(Conv2D(64, (3,3), input_shape = input_shape, name='block1_conv1'))
model.add(Activation("relu"))
model.add(Conv2D(64, (3,3), name='block1_conv2'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool1'))

# Block 2
model.add(Conv2D(128, (3,3), name='block2_conv1'))
model.add(Activation("relu"))
model.add(Conv2D(128, (3,3), name='block2_conv2'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool1'))

# Block 3
model.add(Conv2D(256, (3,3), name='block3_conv1'))
model.add(Activation("relu"))
model.add(Conv2D(256, (3,3), name='block3_conv2'))
model.add(Activation("relu"))
model.add(Conv2D(256, (3,3), name='block3_conv3'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool1'))

# Block 4
model.add(Conv2D(512, (3,3), name='block4_conv1'))
model.add(Activation("relu"))
model.add(Conv2D(512, (3,3), name='block4_conv2'))
model.add(Activation("relu"))
model.add(Conv2D(512, (3,3), name='block4_conv3'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool1'))

# Flatten
model.add(Flatten())
model.add(Dense(1024))
model.add(LeakyReLU(0.3))
model.add(Dense(1024))
model.add(LeakyReLU(0.3))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

with open('x_train_95.pkl', 'rb') as f:
    x_train = pickle.load(f)
    
with open('y_train_95.pkl', 'rb') as f:
    y_train = pickle.load(f)
    
with open('x_test_95.pkl', 'rb') as f:
    x_test = pickle.load(f)
    
with open('y_test_95.pkl', 'rb') as f:
    y_test = pickle.load(f)
    
x_train_max = np.max(x_train)
x_test_max = np.max(x_test)
x_max = max(x_train_max, x_test_max)

x_train_min = np.min(x_train)
x_test_min = np.min(x_test)
x_min = min(x_train_min, x_test_min)

x_train = (x_train - x_min + 0.0) / (0.0 + x_max - x_min)
x_test = (x_test - x_min + 0.0) / (0.0 + x_max - x_min)
    
history = model.fit(x_train, y_train, batch_size=64, epochs = 1000, validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=64)
print(score)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('model_simpler.h5')
print("Model has been saved.")