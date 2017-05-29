import numpy as np
from keras.models import Model, load_model
import keras.backend as K
from keras.applications import vgg16
from scipy.misc import imshow, imsave
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

layeridx = 5
fn = K.function(model.inputs, [model.layers[layeridx].output])

with open('x_train_95.pkl', 'rb') as f:
    x_train = pickle.load(f)
    
imidx = 0
imgs = fn([[x_train[imidx]]])
# print(len(imgs)) # 1
# print(type(imgs[0])) # np.ndarray
# print(imgs[0].shape) # (1, 128, 128, 64)
img = imgs[0][0]
print(img.shape)
for idx in range(img.shape[2]):
    x = img[:, :, idx]
    x = deprocess_image(x)
    imsave('./FeatureMaps/Train{0}/Layer{1}/conv{2}_map{3}.png'.format(imidx, layeridx, layeridx, idx), img[:, :, idx])
    
    