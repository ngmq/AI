# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
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

## print(X.shape)

""" Randomly assign weights. No biases is used """
w = np.random.rand(2, 1)
nw = np.linalg.norm(w)
w = w / (nw + 1e-15)

""" Set learning rate and number of iteration """
lr = 0.001
niter = 6905

all_var = []
last_var = 0 ## Variable to check if variance is increasing

for iter in range(niter):
    y = np.dot(X, w) ## [n, 2] * [2, 1] = [n, 1]
    delta = np.dot(y.T, X) ## [1, n] * [n, 2] = [1, 2]
    delta /= n
    delta = delta.reshape(w.shape)
    assert delta.shape == w.shape # For mistake 1
    
    w = w + lr * delta
    nw = np.linalg.norm(w)
    w = w / (nw + 1e-15)
    
    y2 = np.dot(X, w)
    var = np.sum(y2**2) / n
    all_var.append(var)
    if(var < last_var):
        print(iter, "WEIRD!") # For mistake 2. Now I realized the whole thing is just gradient descent
    last_var = var
    
print("Final weight = ", w)
print(var)

""" Compare the variance on W to the variance explained by the first component in PCA """
pca = PCA(n_components=1)
pca.fit(X)
print(pca.explained_variance_[0])


plt.plot(range(niter), all_var, 'b-')
plt.show()

# plt.plot(x1, x2, 'ro')
# plt.show()