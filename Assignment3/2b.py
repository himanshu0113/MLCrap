import os
import struct
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def read(dataset = "training", path = "."):
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

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


training_data=list(read(dataset='training',path='Data'))
testing_data = list(read(dataset='testing',path='Data'))

np.random.shuffle(training_data)
#print 'training data', training_data

label, pixel = zip(*training_data)
label = np.array(label)
pixel = np.array(pixel)
test_label, test_pixel = zip(*testing_data)
test_label = np.array(test_label)
test_pixel = np.array(test_pixel)

'''
lrate = [0.001, 0.01, 0.1, 0.8, 1]
finalacc = []
for lr in lrate:
    mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(100, 50), random_state=1, verbose=False, early_stopping=True, learning_rate_init= lr)
    mlp.fit(pixel.reshape((pixel.shape[0], -1)), label)
    acc = mlp.score(test_pixel.reshape((test_pixel.shape[0], -1)), test_label)
    finalacc.append(acc)
    print 'Accuracy with learning rate ', str(lr), ': ', acc

'''
#output = mlp.predict_proba(test_pixel.reshape((test_pixel.shape[0], -1)))
#predicted = np.array(np.argmax(output, axis=1))
#print predicted
#acc2 = accuracy_score(test_label, predicted)
#print 'Predicted', mlp.predict(test_pixel.reshape((test_pixel.shape[0], -1)))
#print 'Accuracy: ', acc
#print 'Accracy with Softmax: ', acc2

'''
tot = []
itr = [5, 10, 20, 30]
for i in itr:
    mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(100, 50), verbose=True)
    mlp.fit(pixel.reshape((pixel.shape[0], -1)), label)
    clf = mlp
    acc = mlp.score(test_pixel.reshape((test_pixel.shape[0], -1)), test_label)
    print 'Score:::', acc
    tot.append(acc)

fig = plt.figure(figsize=(10, 10))
plt.xticks(range(len(itr)), itr)
plt.plot(range(len(itr)), tot)
plt.title('Epochs vs Accuracy for MNIST')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('Plots/2b_EpochsvsAcc.png')
plt.show()
'''

wpath = 'Weights/' + '2B' + '_' + 'softmax' + '.pkl'
#joblib.dump(clf, wpath)

model = joblib.load(wpath)
print 'Result', model.score(test_pixel.reshape((test_pixel.shape[0], -1)), test_label)