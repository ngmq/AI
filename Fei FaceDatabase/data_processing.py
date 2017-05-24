import numpy as np
import pickle
import os
from glob import glob
from tqdm import tqdm
from scipy import misc


labels = dict()

with open('all_gender.txt') as f:
    for line in f:
        a = line.replace(':', '').split()
        a = [x.split('-') for x in a]
        low = int(a[0][0])
        high = int(a[0][1])
        
        if(a[1][0] == 'm'):
            gender = 1
        else:
            gender = 0
            
        for id in range(low, high + 1):
            labels[id] = gender            
            
with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
    
all_images = glob('./all_images/*.jpg')

X = list()
Y = list()
for img_path in tqdm(all_images):
    img = misc.imread(img_path)
    img = misc.imresize(img, (128, 128))
    X.append(img)
    
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    id = img_name.split('-')[0]
    id = int(id)
    Y.append(labels[id])
    
print("Read all images done. Saving...")

X = np.array(X)
print(X.shape)

with open("X.pkl", "wb") as f:
    pickle.dump(X, f)
    
print("Save X done.")

Y = np.array(Y)
print(Y.shape)
with open("Y.pkl", "wb") as f:
    pickle.dump(Y, f)
    
print("Save Y done.")

        