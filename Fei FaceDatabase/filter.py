import numpy as np
from keras.models import Model, load_model
import keras.backend as K
from keras.applications import vgg16
from scipy.misc import imshow, imsave
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import pickle

img_width = 128
img_height = 128

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

model = load_model('model.h5')
model.summary()

w = model.layers[1].get_weights()
print(type(w))
print(w.shape)
plt.imshow(w)


