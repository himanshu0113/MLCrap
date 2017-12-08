import numpy as np
import h5py
import random
import os
import struct
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(x):
    return np.exp(x)/(np.sum(np.exp(x)))

class NN(object):

    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

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

    def update_batch(self, mini_batch, eta):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_b, delta_w = self.backpropogation(x, y)
            #print 'delta W', len(new_w), new_w
            #print 'delta B', len(new_b), new_b
            new_b = [b + b_bar for b, b_bar in zip(delta_b, new_b)]
            new_w = [w + w_bar for w, w_bar in zip(delta_w, new_w)]

        self.biases = [b - ((eta/len(mini_batch))*delta_b) for b, delta_b in zip(self.biases, new_b)]
        self.weights = [w - ((eta/len(mini_batch)*delta_w)) for w, delta_w in zip(self.weights, new_w)]

    def forward_feed(self, x):

        a = x.reshape((784,1))
        activations = []
        zs = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            activations.append(a)
            zs.append(z)

        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = softmax(z)
        activations.append(a)
        zs.append(z)

        return activations, zs


    def backpropogation(self, x, y):
        #print 'x', x.shape
        #print 'y', y.shape
        x = np.array(x).reshape((len(x), 1))
        y = np.array(y).reshape((len(y), 1))
        #print 'shapes', x.shape, y.shape

        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]

        #forward
        activations, zs = self.forward_feed(x)
        activations = [x] + activations

        #backward
        sp = (sigmoid(zs[-1]) * (1 - sigmoid(zs[-1])))
        sp = np.array([np.array(s) for s in sp])
        delta = (activations[-1] - y)
        delta = np.array(delta)

        new_b[-1] = delta
        new_w[-1] = np.dot(delta, activations[-2].T)

        #print len(activations)t
        for i in range(2, self.num_layers):
            sp = (sigmoid(zs[-i])*(1 - sigmoid(zs[-i])))
            #sp = relu_prime(zs[-i])
            delta = np.dot(self.weights[-i+1].T, delta)*sp
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

        Y_bar = [[0]*10 for _ in range(len(result))]
        for i in range(len(result)):
            j = np.argmax(result[i])
            Y_bar[i][j] = 1

        #print 'Y_bar', Y_bar
        #print 'Y', Y
        return np.sum(int(x==x_bar.tolist()) for x, x_bar in zip(Y_bar, Y))


def load_ubyte(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (img[idx], lbl[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

training_data=list(load_ubyte(dataset='training', path='Data'))
testing_data = list(load_ubyte(dataset='testing', path='Data'))

np.random.shuffle(training_data)

pixel, label = zip(*training_data)
label = np.array(label)
pixel = np.array(pixel)
pixel = pixel.reshape((pixel.shape[0], -1))
pixel = np.divide(pixel, 255.0)

label_new = np.zeros((len(label), 10))
for i in range(len(label)):
    label_new[i][label[i]] = 1

label = label_new

#print 'pixel', pixel.shape
#print 'label', label.shape, label

test_pixel, test_label = zip(*testing_data)
test_label = np.array(test_label)
test_pixel = np.array(test_pixel)
test_pixel = test_pixel.reshape((test_pixel.shape[0], -1))
test_pixel = np.divide(test_pixel, 255.0)

label_new = np.zeros((len(test_label), 10))
for i in range(len(test_label)):
    label_new[i][test_label[i]] = 1

test_label = label_new

#print 'Input', pixel, label

#net = NN([784, 100, 50, 10])
#net.run(training_data=zip(pixel, label), test_data=zip(test_pixel, test_label), eta=0.001, epochs=30)
#clf = net

wpath = 'Weights/' + '1B' + '_' + 'softmax' + '.pkl'
#joblib.dump(clf, wpath)

model = joblib.load(wpath)
print 'Result', model.predict(test_data=zip(test_pixel, test_label)) / float(len(test_pixel))