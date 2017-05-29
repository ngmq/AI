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
    
vgg16 = VGG16(include_top=False, weights=None, input_shape=input_shape)
x = Flatten()(vgg16.layers[-1].output)
# x = Dense(1024)(x)
# x = LeakyReLU(0.3)(x)
# x = Dense(1024)(x)
# x = LeakyReLU(0.3)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(input = vgg16.input, output = x)

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

with open('x_train_80.pkl', 'rb') as f:
    x_train = pickle.load(f)
    
with open('y_train_80.pkl', 'rb') as f:
    y_train = pickle.load(f)
    
with open('x_test_80.pkl', 'rb') as f:
    x_test = pickle.load(f)
    
with open('y_test_80.pkl', 'rb') as f:
    y_test = pickle.load(f)
    
history = model.fit(x_train, y_train, batch_size=32, epochs = 40, validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('Figure1.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('Figure2.png')

model.save('model.h5')
print("Model has been saved.")
