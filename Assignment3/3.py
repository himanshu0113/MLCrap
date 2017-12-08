import h5py
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        X = X.reshape((X.shape[0], -1))
        Y = hf['Y'][:]
	return X, Y

K = 5
def evaluate_mlp(X, Y, i):
    mlp = MLPClassifier(hidden_layer_sizes=(500, 300, 50), activation='logistic', verbose=False, early_stopping=True, max_iter=i)
    validations = []
    for k in range(K):
        X_train = list(X)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        Y_train = list(Y)
        Y_test = Y_train.pop(k)
        Y_train = np.concatenate(Y_train)
        mlp.fit(X_train, Y_train)
        #predicted = mlp.predict(X_test)
        accuracy = mlp.score(X_test, Y_test)
        #plt.plot(mlp.validation_scores_)
        validations.append(accuracy)
    validation = sum(validations)/float(len(validations))
    clf = mlp
    print 'Accuracy: ', validation
    return clf

X, Y  = load_h5py('Data/dataset_partA.h5')

d = zip(X, Y)
np.random.shuffle(d)
X, Y = zip(*d)
X = list(X)
Y = list(Y)

X_folds = np.array_split(X, K)
Y_folds = np.array_split(Y, K)

#fig = plt.figure(figsize=(10, 10))

#for i in [20, 50, 80, 100]:
    #mlp = MLPClassifier(hidden_layer_sizes=(500, 300, 50), activation='logistic', verbose=True, early_stopping=False, max_iter=20)
    #mlp.fit(X, Y)
    #print 'Score : ', mlp.score()


#plt.title('Epochs vs Accuracy for MNIST (subset)')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.plot(mlp.validation_scores_)
#plt.savefig('Plots/2a_epochVSaccuracy.png')
#plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,  shuffle = True)

clf = evaluate_mlp(X_folds, Y_folds, 50)

wpath = 'Weights/' + '3' + '_' + 'logistic(500,300,50)' + '.pkl'
joblib.dump(clf, wpath)

