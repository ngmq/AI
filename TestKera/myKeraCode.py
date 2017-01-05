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