import os
import os.path
import argparse
from sklearn.manifold import TSNE
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

X, Y = load_h5py(args.data)

X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, n_iter=5000).fit_transform(X)

dataset_name = args.data.split('/')

if(dataset_name[-1] == "part_A_train.h5"):
	colorset=['red','orange','pink','green','black','gray','blue','brown','yellow','cyan']
else:
	colorset = ['orange', 'blue']

color=[]
for row in Y:
    pos = row.tolist().index(1)
    color.append(colorset[pos])


#plot_name = args.plots_save_dir + dataset_name[-1] + '.png'
fig = plt.figure(figsize=(10,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = color)
#plt.savefig(plot_name)
plt.show()
