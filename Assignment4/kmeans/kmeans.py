import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics

import data_load as dl

def eucl_dist(a, b):
    return np.linalg.norm(a - b, axis=1)

def kmeans(X, Y, k):

    #random centroids
    m, n= np.shape(X)
    #print m, n
    C = np.mat(np.zeros((k, n)))
    label = np.mat(np.zeros((m, 2)))

    C[:, range(0,n)] = X[np.random.choice(100, k, replace = False), :]
    #print C

    # copy of initial clusteres assigned
    C_old = C.copy()
    cluster_update = True
    n_iter = 0
    objective = []

    # Running until cluster updation stops
    while(cluster_update):
        cluster_update = False

        for i in range(m):
            min_dist = np.inf
            min_index = -1

            for j in range(k):
                dist_ij = eucl_dist(X[i,:], C[j,:])
                if dist_ij<min_dist:
                    min_dist = dist_ij
                    min_index = j
                    #print min_index

            if label[i,0] != min_index:
                cluster_update = True

            # **2 is for normalizing the distance
            label[i,:] = min_index, min_dist**2

        for i in range(k):
            points = X[np.nonzero(label[:, 0].A == i)[0]]
            C[i, :] = np.mean(points, axis=0)

        # value of objective function
        s = 0
        for i in range(m):
            s += np.square(label[i,1])

        objective.append(s)
        n_iter = n_iter + 1

    return C, C_old, label, n_iter, objective


#main

X, Y = dl.load_iris_data()

C, C_old, label, n_iter, objective = kmeans(X,Y, 3)


# metrics
for k in [2, 3, 12]:
    ari = 0
    nmi = 0
    ami = 0
    for i in range(5):
        C, C_old, label, n_iter, objective = kmeans(X, Y, 3)
        y_pred = [int(i) for i in label[:,0]]
        ari += metrics.adjusted_rand_score(Y, y_pred)
        nmi += metrics.normalized_mutual_info_score(Y, y_pred)
        ami += metrics.adjusted_mutual_info_score(Y, y_pred)
    ari = np.mean(ari)
    nmi = np.mean(nmi)
    ami = np.mean(ami)
    print 'mean', ari, nmi, ami

# print label
# print X[0]
# print X.shape
# print [int(i) for i in label[:,0]]

"""
#plotting
X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, n_iter=1000, random_state=100).fit_transform(X)
colorset=['blue','orange','green','pink','black','gray','red','brown','yellow','cyan']

color=[]
for row in [int(i) for i in label[:,0]]:
    color.append(colorset[row])

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = color)
plt.title('Visualization of Vertebral dataset after K-means')
plt.savefig('kmeans/plots/vertebral_dataset_kmeans.png')
plt.show()
"""

"""
plt.plot(range(len(objective)), objective)
plt.title('Objective vs Iteration Number: Seed Dataset')
plt.savefig('seed_dataset_obj.png')
plt.show()

"""