
#loading functions for all the datasets

import pandas as pd
import numpy as np

# Load the test data
def load_seed_data(data_path = 'kmeans/data/seed_data/seeds_dataset.txt'):
    X = pd.read_csv(data_path, header=None, delim_whitespace=True, usecols=range(0,7))
    Y = pd.read_csv(data_path, header=None, delim_whitespace=True, usecols=[7])
    X = np.array(X)
    _, Y = np.unique(np.array(Y), return_inverse=True)
    return X, Y

def load_iris_data(data_path = 'kmeans/data/iris_data/iris.data.txt'):
    X = pd.read_csv(data_path, header=None, usecols=range(0,4))
    Y = pd.read_csv(data_path, header=None, usecols=[4])
    X = np.array(X)
    _, Y = np.unique(np.array(Y), return_inverse=True)
    return X, Y

def load_vertebral_data(data_path = 'kmeans/data/vertebral_column_data/column_3C.dat'):
    X = pd.read_csv(data_path, header=None, usecols=range(0,6), delim_whitespace=True,)
    Y = pd.read_csv(data_path, header=None, usecols=[6], delim_whitespace=True,)
    X = np.array(X)
    _, Y = np.unique(np.array(Y), return_inverse=True)
    #X = normalize_data(X)
    mean = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    X = np.array([(x,y) for x,y in zip(X,Y) if not (np.any(x < mean - 2*sd) or np.any(x > mean + 2*sd))])
    X, Y = zip(*X)
    X = normalize_data(X)
    #X = np.delete(X, 4, axis=1)         #verify
    return X, Y

def load_segmentation_data(data_path = 'kmeans/data/image_data/segmentation.data.txt'):
    X = pd.read_csv(data_path, header=None, usecols=range(1,20), skiprows=5)
    Y = pd.read_csv(data_path, header=None, usecols=[0], skiprows=5)
    X = np.array(X)
    _, Y = np.unique(np.array(Y), return_inverse=True)
    X = normalize_data(X)
    X = np.delete(X, [2,3,4], axis=1)
    return X, Y

def normalize_data(X):
    mx = np.max(X, axis=0)
    mn = np.min(X, axis=0)
    l, _ = np.shape(X)
    mx = np.tile(mx, (l, 1))
    mn = np.tile(mn, (l, 1))

    norm_data = np.divide(np.subtract(X,mn), np.subtract(mx, mn))
    return norm_data

