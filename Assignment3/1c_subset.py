import numpy as np
import h5py
import random
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def sigmoid_1(z):
    return .5 * (1 + np.tanh(.5 * z))

def sigmoid_2(z):
    res = []
    for x in z:
        if x < 0:
            ans = 1.0 / (1.0 + np.exp(x))
        else:
            ans = 1.0 / (1.0 + np.exp(-x))

        res.append(ans.tolist())
    #print 'res::::', res
    return np.array(res)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(x):
    return np.exp(x)/(np.sum(np.exp(x)))

def relu(x):
    x[x<0] = 1
    return np.log(x)

def relu_prime(z):
    res = []
    for x in z:
        if x > 0:
            ans = [1.0]
        else:
            ans = [0.0]

        res.append(ans)
    # print 'res::::', res
    return np.array(res)

class NN(object):

    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        #print 'Weights', self.weights

    def run(self, training_data, epochs = 30, batch_size = 10, eta = 0.8, test_data=None):
        train_len = len(training_data)
        test_len = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size]
                for k in xrange(0, train_len, batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.predict(test_data), test_len)
            else:
                print "Epoch {0} complete".format(j)

            #print 'delta W', len(self.weights), self.weights
            #print 'delta B', len(self.biases), self.biases

    def update_batch(self, mini_batch, eta):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_b, delta_w = self.backpropogation(x, y)
            #print 'delta W', len(new_w), new_w
            #print 'delta B', len(new_b), new_b
            new_b = [b + b_bar for b, b_bar in zip(delta_b, new_b)]
            new_w = [w + w_bar for w, w_bar in zip(delta_w, new_w)]

        #print 'delta W', len(new_w), new_w
        #print 'delta B', len(new_b), new_b

        temp = self.weights
        self.biases = [b - ((eta/len(mini_batch))*d_b) for b, d_b in zip(self.biases, new_b)]
        self.weights = [w - ((eta/len(mini_batch)*d_w)) for w, d_w in zip(self.weights, new_w)]

        #print 'diff', [x-y for x, y in zip(temp, self.weights)]

    def forward_feed(self, x):

        a = x.reshape((784,1))
        activations = []
        zs = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            a = relu(z)
            activations.append(a)
            zs.append(z)

        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = sigmoid(z)
        #print 'a ::::: ', a
        activations.append(a)
        zs.append(z)

        return activations, zs


    def backpropogation(self, x, y):
        x = np.array(x).reshape((len(x), 1))
        y = np.array(y).reshape((len(y), 1))
        #print 'shapes', x.shape, y.shape

        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]

        #forward
        activations, zs = self.forward_feed(x)
        #print 'z', len(zs), zs
        #print 'acti', len(activations[0]), len(activations[1]), activations
        activations = [x] + activations

        #backward
        sp = (sigmoid(zs[-1]) * (1 - sigmoid(zs[-1])))
        sp = np.array([np.array(s) for s in sp])
        delta = (activations[-1] - y) * sp
        #print 'delta', delta
        delta = np.array(delta)

        new_b[-1] = delta
        new_w[-1] = np.dot(delta, activations[-2].T)

        #print len(activations)
        for i in range(2, self.num_layers):
            sp = relu_prime(zs[-i])
            #print 'sp', sp
            #print 'delta_pre', delta
            delta = np.dot(self.weights[-i+1].T, delta) * sp
            #print 'delta', delta
            #print 'acti', activations[-i-1]
            new_b[-i] = delta
            new_w[-i] = np.dot(delta, activations[-i-1].T)

        return new_b, new_w

    def predict(self, test_data):
        X = []
        Y = []
        for x, y in test_data:
            X.append(x)
            Y.append(y)
        #print 'X', len(X)
        #print 'Y', len(Y), Y
        result = []
        for x, y in test_data:
            a, z = self.forward_feed(x)
            #print 'a[-1]', a[-1]
            A = []
            for k in a[-1]:
                #print 'k', k
                A.append(k[0])
            #print 'A', A
            result.append(A)

        #print 'result', result

        Y_bar = [[0,0] for _ in range(len(result))]
        for i in range(len(result)):
            j = np.argmax(result[i])
            Y_bar[i][j] = 1

        #print 'Y_bar', Y_bar
        #print 'Y', Y
        return np.sum(int(x==x_bar.tolist()) for x, x_bar in zip(Y_bar, Y))


def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        X = X.reshape((X.shape[0], -1))
        Y = hf['Y'][:]
	return X, Y


def evaluate_model(X, Y, e):
    acc = []
    for k in range(K):
        X_train = list(X)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        Y_train = list(Y)
        Y_test = Y_train.pop(k)
        Y_train = np.concatenate(Y_train)
        net.run(training_data=zip(X_train, Y_train), test_data=zip(X_test, Y_test), eta=0.001, epochs=e)
        accuracy = net.predict(zip(X_test, Y_test)) / float(len(X_test))
        acc.append(accuracy)
    finalacc = sum(acc) / float(len(acc))
    print 'Accuracy with epochs ', str(e), ': ', finalacc
    return finalacc


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

K = 3

X_folds = np.array_split(X, K)
Y_folds = np.array_split(Y, K)

net = NN([784, 100, 50, 2])

epochs = [10, 30, 50, 80]
kfoldacc = []
#for e in epochs:
#    kfoldacc.append(evaluate_model(X_folds, Y_folds, e))

'''
plt.figure(figsize=(10, 10))
plt.title('Epochs vs Accuracy for MNIST (Subset)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(len(epochs)), epochs)
plt.plot(range(len(epochs)), kfoldacc)
plt.savefig('Plots/1C_EpochsvsAcc_ReLU_subset.png')
plt.show()
'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,  shuffle = True)

wpath = 'Weights/' + '1C' + '_' + 'relu' + '.pkl'
#joblib.dump(clf, wpath)

model = joblib.load(wpath)
print 'Result', model.predict(test_data=zip(X_test, Y_test)) / float(len(X_test))