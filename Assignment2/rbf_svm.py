import os
import os.path
import argparse
from sklearn.manifold import TSNE
import h5py
import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.metrics import mean_absolute_error as mae
#from mpl_toolkits.mplot3d import Axes3D

"""
parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()
"""
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


def radial_basis(x, y, gamma=1):
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

def rbf_predict(X_test, X_train, Y_train):
    #One-vs-Rest
    out = []
    output = []
    coeff = []
    sv = []
    interc =[]
    
    for i in range(len(np.unique(Y_train))):
        #loop for each class
        Y = [1 if y==i else 0 for y in Y_train]
        print "predict" + str(i)
        clf = svm.SVC(kernel =proxy_kernel).fit(X_train,Y)
        coeff.append(clf.dual_coef_)
        #print len(coeff[i])
        sv.append(clf.support_)
        #print len(sv[i])
        interc.append(clf.intercept_)
        #print len(interc)
        
        
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

def run(i,X,Y, color):
    k = int(len(X)*0.8)
    
    X_train = X[:k]
    Y_train = Y[:k]
    X_test = X[k+1:]
    Y_test = Y[k+1:]
    
    #clf = svm.SVC(kernel =proxy_kernel).fit(X_train,Y_train)
    
    Z = linear_predict(X_test, X_train, Y_train)
    
    error = mae(Y_test, Z)
    print 'Error for dataset ' + str(i) 
    print error
    

#-------------
    
run(1, X1, Y1, color1)
run(2, X2, Y2, color2)
run(3, X3, Y3, color3)
run(4, X4, Y4, color4)
run(5, X5, Y5, color5)