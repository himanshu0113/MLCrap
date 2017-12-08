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


def evaluate_mlp(X, Y, alpha):
    acc = []
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='logistic', verbose=False, early_stopping=True, learning_rate_init=alpha)
    for k in range(K):
        X_train = list(X)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        Y_train = list(Y)
        Y_test = Y_train.pop(k)
        Y_train = np.concatenate(Y_train)
        mlp.fit(X_train, Y_train)
        accuracy = mlp.score(X_test, Y_test)
        acc.append(accuracy)
    finalacc = sum(acc)/float(len(acc))
    print 'Accuracy with learning rate ', str(alpha), ': ', finalacc
    return finalacc

X, Y  = load_h5py('Data/dataset_partA.h5')

d = zip(X, Y)
np.random.shuffle(d)
X, Y = zip(*d)
X = list(X)
Y = list(Y)

K = 3
alphas = [0.001]


X_folds = np.array_split(X, K)
Y_folds = np.array_split(Y, K)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='logistic', verbose=True, early_stopping=True)
mlp.fit(X, Y)

'''

kfoldacc = []
for a in alphas:
    kfoldacc.append(evaluate_mlp(X_folds, Y_folds, a))

fig = plt.figure(figsize=(10, 10))
plt.figure(figsize=(10, 10))
plt.title('Hyperparameter vs Accuracy for MNIST (subset)')
plt.xlabel('Hperparameter')
plt.ylabel('Accuracy')
plt.xticks(range(len(alphas)), alphas)
plt.plot(range(len(alphas)), kfoldacc)

#plt.savefig('Plots/2AEpochvsAcc.png')
#plt.show()
'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,  shuffle = True)

wpath = 'Weights/' + '2A' + '_' + 'sigmoid' + '.pkl'
#joblib.dump(mlp, wpath)

model = joblib.load(wpath)
print 'Result', model.score(X_test, Y_test)