import os
import os.path
import struct
from sklearn.manifold import TSNE
import h5py
import numpy as np
import matplotlib.pyplot as plt

"""
parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

"""

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

#X, Y = load_h5py(args.data)
#X, Y = load_h5py('Data/dataset_partA.h5')
#print X.shape, Y.shape
#print Y[:100]

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

colorset=['blue','orange', 'red', 'yellow', 'black', 'gray', 'pink', 'cyan', 'green', 'brown']

color = []
for label in label:
	color.append(colorset[label])

X_tsne = TSNE(n_components=2, perplexity=20, learning_rate=10, n_iter=300).fit_transform(pixel)

fig = plt.figure(figsize=(10, 10))
plt.title('MNIST Visualization')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = color)
plt.savefig('Plots/Complete_visualization.png')
plt.show()
