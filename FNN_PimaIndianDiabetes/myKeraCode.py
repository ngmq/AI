from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input(X) and output(Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

print type(dataset) # numpy.ndarray

#print X
#print "Y = "
#print Y

# Create network model
model = Sequential()
model.add(Dense(12, input_dim = 8, init = 'uniform', activation = 'relu'))
model.add(Dense(8, init = 'uniform', activation = 'relu'))
model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

# Compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fit model 
model.fit(X, Y, nb_epoch = 150, batch_size = 10)

# Evaluate model 
scores = model.evaluate(X, Y)

print model.metrics_names
print scores
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )