#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import Callback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


n_points = 200
x1 = np.linspace(0, 2, n_points)
y = 5*np.random.rand(n_points)
x2 = (y - 1 - 2 * x1) / 3
# y = 2*x1 + 3*x2 + 1
x = np.array([ [x1[i], x2[i] ] for i in range(n_points) ])
#y = np.array([0] * int(n_points / 2) + list(x[:int(n_points / 2)])) * 2

#plt.figure(figsize=(5, 2))
#plt.plot(x, y, linewidth=2)
#plt.title('ridiculously simple data')
#plt.xlabel('a')
#plt.ylabel('b')
#plt.show()

np.random.seed(233)
model = Sequential()
# First layer
model.add(Dense(output_dim = 2, input_dim = 2, init="uniform"))
model.add(Activation("linear"))

# Second layer
model.add(Dense(output_dim = 1, init="uniform"))
model.add(Activation("linear"))

# compile model
model.compile(loss='mean_squared_error', optimizer='sgd')

# print initial weights
weights = model.layers[0].get_weights()
#print(weights.shape)
print(weights)
w0 = weights[1][0]
w1 = weights[0][0][0]
w2 = weights[0][1][0]

print('neural net initialized with weights w0 = %.2f, w1 = %.2f, w2 = %.2f' %(w0,w1,w2))

class TrainingHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.predictions = []
		self.i = 0
		self.save_every = 50

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.i += 1
		#if self.i % self.save_every == 0:
			#pred = model.predict(X_train)
			#self.predictions.append(pred)

history = TrainingHistory()

X_train = np.array(x, ndmin = 2)
Y_train = np.array(y, ndmin = 2).T
print(X_train.shape)
print(y.shape)
print(Y_train.shape)
model.fit(X_train, 
          Y_train,
	  nb_epoch = 2000,
	  verbose = 0,
	  batch_size = 200,
	  callbacks=[history])

weights = model.layers[0].get_weights()
w0 = weights[1][0]
w1 = weights[0][0][0]
w2 = weights[0][1][0]

print('neural net weights after training w0 = %.2f, w1 = %.2f, w2 = %.2f' %(w0,w1,w2))

plt.figure(figsize=(6, 3))
plt.plot(history.losses)
plt.ylabel('error')
plt.xlabel('iteration')
plt.title('training error')
plt.show()

x_test = np.array([[1, 2]]) #np.random.rand(100)
x_test = np.random.rand(50, 2)
y_test = 2*x_test[:, 0] + 3*x_test[:, 1] + 1
y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
print("Loss = %.2f" % loss)


