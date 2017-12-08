import h5py
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import numpy as np

#from 1a import NN

# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

X, Y  = load_h5py('Data/dataset_partA.h5')

d = zip(X, Y)
np.random.shuffle(d)
X, Y = zip(*d)
X = list(X)
Y = list(Y)

X = np.divide(X, 255.0)
Y_new = np.zeros((len(Y), 2))
for i in range(len(Y)):
    if Y[i] == 7:
        Y_new[i][0] = 1
    else:
        Y_new[i][1] = 1
Y = Y_new


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,  shuffle = True)

model = joblib.load('Weights/1A_sigmoid.pkl') 	#Specify the file to load

print 'Result', model.predict(test_data=zip(X_test, Y_test))
