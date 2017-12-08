import os
import os.path
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import data_load as dl

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

#X, Y = dl.load_iris_data(args.data)
X, Y = dl.load_vertebral_data()

print X.shape

X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, n_iter=1000, random_state=100).fit_transform(X)

dataset_name = args.data.split('/')

colorset=['blue','orange','green','pink','black','gray','red','brown','yellow','cyan']

color=[]
for row in Y:
    color.append(colorset[row])


plot_name = args.plots_save_dir + dataset_name[-1] + '1.png'
plt.title('Visualization of ' + dataset_name[-1])
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = color)
plt.savefig(plot_name)
plt.show()
