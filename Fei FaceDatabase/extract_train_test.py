import numpy as np
import pickle

print("Loading X and Y...")
with open('X.pkl', 'rb') as f:
    X = pickle.load(f)
    
with open('Y.pkl', 'rb') as f:
    Y = pickle.load(f)
    
train_size = 80 * 14
n_male = 0
n_female = 0
X_train = list()
Y_train = list()
X_test = list()
Y_test = list()
last_id = 0

print("Extracting train and test set...")
for i, img in enumerate(X):
    # print i, Y[i], img.shape
    if(Y[i] == 0):
        if(n_female < train_size):
            X_train.append(img)
            Y_train.append(Y[i])
        else:
            X_test.append(img)
            Y_test.append(Y[i])
        n_female += 1
    else:
        if(n_male < train_size):
            X_train.append(img)
            Y_train.append(Y[i])
        else:
            X_test.append(img)
            Y_test.append(Y[i])
        n_male += 1

X_train = np.array(X_train)       
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
 
with open('X_train_80.pkl', 'wb') as f:
    pickle.dump(X_train, f)
    
with open('Y_train_80.pkl', 'wb') as f:
    pickle.dump(Y_train, f)
    
with open('X_test_80.pkl', 'wb') as f:
    pickle.dump(X_test, f)
    
with open('Y_test_80.pkl', 'wb') as f:
    pickle.dump(Y_test, f)