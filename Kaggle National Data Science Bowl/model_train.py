from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

nb_classes = 3

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=150,
        shear_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'trivial_train',
        target_size=(80, 80),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'trivial_test',
        target_size=(80, 80),
        batch_size=32,
        class_mode='categorical')

if K.image_data_format() == "channels_last":
    input_shape = (80, 80, 3)
else:
    input_shape = (3, 80, 80)
    
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('tanh'))
model.add(Dropout(0.7))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# 

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=30,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=30)
        
model.save('firstModel.h5')