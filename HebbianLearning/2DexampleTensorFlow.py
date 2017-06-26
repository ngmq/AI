import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

np.random.seed(1991)

""" Generate randomly 33 data points in 2D and centrelize them (i.e. mean = 0) """
n = 33
x1 = np.random.rand(n)
x2 = -5*(x1) + 3 + 1.5*np.random.rand(n)

print(x1)
print(x2)

X = np.array([x1, x2])
X = X.T
X = X - np.mean(X, axis=0)

""" Randomly assign weights. No biases is used """
w = np.random.rand(2, 1)
nw = np.linalg.norm(w)
w = w / (nw + 1e-15)

""" Set learning rate and number of iteration """
lr = 0.001
niter = 6905
all_var = []

""" Calculate with Tensorflow """

input = tf.placeholder(tf.float64, [None, 2], name='input')
weight = tf.Variable(w, dtype=tf.float64, name='weight', trainable = True)

def proj_variance(x, w):
    # w = tf.nn.l2_normalize(w, dim=0) # can also normalize weight in here, but better do it outside scope of loss function
    y = tf.matmul(x, w)
    y = tf.square(y)
    y = tf.reduce_sum(y)
    y = tf.divide(y, 33)
    y = tf.negative(y)
    return y
    
loss = proj_variance(input, weight)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
normalize_weight = tf.assign(weight, tf.nn.l2_normalize(weight, dim=0))
print(normalize_weight)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for iter in range(niter):
    sess.run(optimizer, feed_dict = {input: X})
    sess.run(normalize_weight)    
    var = loss.eval(feed_dict = {input: X})
    all_var.append(-var)
sess.close()
print(-var)

""" Compare the variance on W to the variance explained by the first component in PCA """
pca = PCA(n_components=1)
pca.fit(X)
print(pca.explained_variance_[0])

plt.plot(range(niter), all_var, 'b-')
plt.show()