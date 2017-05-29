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
count = 0
toTrain = 0
toTest = 0

testid = [9, 12, 107, 111, 138, 195, 349, 402, 403, 404, 447, 527]

for i, img in enumerate(X):
    # print Y[i]
    # if(Y[i] == 0):
        # if(n_female < train_size):
            # X_train.append(img)
            # Y_train.append(Y[i])
        # else:
            # X_test.append(img)
            # Y_test.append(Y[i])
        # n_female += 1
    # else:
        # if(n_male < train_size):
            # X_train.append(img)
            # Y_train.append(Y[i])
        # else:
            # X_test.append(img)
            # Y_test.append(Y[i])
        # n_male += 1
    if(count < 9):
        X_train.append(img)
        Y_train.append(Y[i])
        toTrain += 1
    else:
        X_test.append(img)
        Y_test.append(Y[i])
        toTest += 1
    count += 1
    count = count % 14
    
print(toTrain)
print(toTest)

# X_train = np.array(X_train)       
# Y_train = np.array(Y_train)
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)
 
# with open('x_train_80.pkl', 'wb') as f:
    # pickle.dump(X_train, f)
    
# with open('y_train_80.pkl', 'wb') as f:
    # pickle.dump(Y_train, f)
    
# with open('x_test_80.pkl', 'wb') as f:
    # pickle.dump(X_test, f)
    
# with open('y_test_80.pkl', 'wb') as f:
    # pickle.dump(Y_test, f)