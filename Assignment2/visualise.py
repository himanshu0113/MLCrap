from sklearn.manifold import TSNE
import h5py
import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm
#from mpl_toolkits.mplot3d import Axes3D

# Load the test data
def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        #print("keys: %s" % hf.keys())
        X = hf['x'][:]
        Y = hf['y'][:]
    return X, Y

X1, Y1= load_h5py("data_1.h5")
X2, Y2= load_h5py("data_2.h5")
X3, Y3= load_h5py("data_3.h5")
X4, Y4= load_h5py("data_4.h5")
X5, Y5= load_h5py("data_5.h5")

colorset=['orange','blue','yellow','black','red','cyan']
color1=[]
for row in Y1:
    color1.append(colorset[row])

color2=[]
for row in Y2:
    color2.append(colorset[row])

color3=[]
for row in Y3:
    color3.append(colorset[row])

color4=[]
for row in Y4:
    color4.append(colorset[row])

color5=[]
for row in Y5:
    color5.append(colorset[row])


def radial_basis(x, y, gamma=0.7):
    return np.exp(-gamma * (np.linalg.norm(np.subtract(x, y)))**2)

def proxy_kernel(X, Y, K=radial_basis):
    """Another function to return the gram_matrix,
    which is needed in SVC's kernel or fit
    """
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = K(x, y)
    return gram_matrix

def rbf_predict(X_test):
    #One-vs-Rest
    out = []
    output = []
    coeff = []
    sv = []
    interc =[]
    
    for i in range(len(np.unique(Y_train))):
        #loop for each class
        Y = [0 if y==i else 1 for y in Y_train]
        print "predict" + str(i)
        clf = svm.SVC(kernel ='rbf').fit(X_train,Y)
        coeff.append(clf.dual_coef_)
        print len(coeff[i])
        sv.append(clf.support_)
        print len(sv[i])
        interc.append(clf.intercept_)
        print len(interc)
        
        
    for j in range(len(np.unique(Y_train))):
        gram_mat = proxy_kernel(X_train[sv[j]],X_test)
        t = np.dot(coeff[j], gram_mat) + interc[j]
        t = np.array(t).flatten()
        out.append(t)
    
    #print len(out)
    
    for i in range(len(X_test)):
        maxo = out[0][i]
        maxo_i = 0
        for j in range(len(out)):
            if maxo<out[j][i]:
                maxo = out[j][i]
                maxo_i = j
        output.append(maxo_i)
        
    return output

X_train = X5
Y_train = Y5
color = color5
h = .02  # step size in the mesh

clf = svm.SVC(kernel ='rbf').fit(X_train,Y_train)

# create a mesh to plot in
x_min, x_max = X_train[:,0].min() - 1, X_train[:,0].max() + 1
y_min, y_max = X_train[:,1].min() - 1, X_train[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig = plt.figure(figsize=(5, 5))

#Z = np.array(rbf_predict(np.c_[xx.ravel(), yy.ravel()]))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(X_train[:,0], X_train[:,1], c = color, cmap=plt.cm.coolwarm)
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.savefig("Plots/withBoundary/data5")
plt.show()