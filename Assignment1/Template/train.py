import os
import os.path
import argparse
import h5py

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.externals import joblib

import numpy as np
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

dataset_name = args.train_data.split('/')
# Weights path
wpath = args.weights_path + dataset_name[-1] + '_' + args.model_name + '.pkl'
# Plot path
ppath = args.plots_save_dir + dataset_name[-1] + '_' + args.model_name + '.png'
# Evaluation Functions

K = 3

def evaluate_LR(X, Y, c):
    scores = list()
    logistic = linear_model.LogisticRegression(C = c, random_state = 50)

    for k in range(K):
        X_train = list(X)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        Y_train = list(Y)
        Y_test = Y_train.pop(k)
        Y_train = np.concatenate(Y_train)
        scores.append(logistic.fit(X_train, Y_train).score(X_test, Y_test))
    #print scores
    return sum(scores)/len(scores)


def evauluate_GNB(X, Y):
    scores = list()
    gnb = GaussianNB()

    for k in range(K):
        X_train = list(X)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        Y_train = list(Y)
        Y_test = Y_train.pop(k)
        Y_train = np.concatenate(Y_train)
        scores.append(gnb.fit(X_train, Y_train).score(X_test, Y_test))
    #print scores
    return sum(scores)/len(scores)

def evauluate_DTC(X, Y, param):
    scores = list()
    dtc = DecisionTreeClassifier(max_depth = param[0], min_samples_split = param[1], random_state = 50)

    for k in range(K):
        X_train = list(X)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        Y_train = list(Y)
        Y_test = Y_train.pop(k)
        Y_train = np.concatenate(Y_train)
        scores.append(dtc.fit(X_train, Y_train).score(X_train, Y_train))
    #print scores
    return sum(scores)/len(scores)


# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

X, Y_ = load_h5py(args.train_data)

# Preprocess data and split it
Y = []
for row in Y_:
    pos = row.tolist().index(1)
        #if pos == 0:                        #verify this approach
    Y.append(pos)
        #else:
#Y.append(1)

X_folds = np.array_split(X, K)
Y_folds = np.array_split(Y, K)
#print X_folds
#print Y_folds


# Train the models

if args.model_name == 'GaussianNB':
    accuracy = []

    score = evauluate_GNB(X_folds, Y_folds)
    print("GaussianNB Score: %.3f" %(score))

    gnb = GaussianNB().fit(np.concatenate(X_folds), np.concatenate(Y_folds))
    joblib.dump(gnb, wpath)

elif args.model_name == 'LogisticRegression':
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    accuracy = []
    #grid search using C parmeter of Logistic Regression
    best_score, best_cfg = float(0), None
    for c in C:
        score = evaluate_LR(X_folds, Y_folds, c)
        accuracy.append(score)
        if score > best_score:
            best_score, best_cfg = score, c
        print("Logistic Regression %s Score: %.3f" %(c, score))

    print("Best possible LR %s Score : %.3f" %(best_cfg, best_score))

    logistic = linear_model.LogisticRegression(C = best_cfg, random_state = 50).fit(np.concatenate(X_folds), np.concatenate(Y_folds))
    joblib.dump(logistic, wpath)

    plt.xticks(range(7), C)
    plt.plot(range(7), accuracy)
    plt.savefig(ppath)
    plt.show()

elif args.model_name == 'DecisionTreeClassifier':
    accuracy = []
    max_depth = range(1,20,2)
    min_samples_split = range(10,500,20)
    param_list = []

    best_score, best_cfg = float(0), None
    for max in max_depth:
        for min in min_samples_split:
            param = (max, min)
            param_list.append(param)
            score = evauluate_DTC(X_folds, Y_folds, param)
            accuracy.append(score)
            if score > best_score:
                best_score, best_cfg = score, param
            print("Decision tree Classifier %s Score: %.3f" %(param, score))
    print("Best possible Decision tree Classifier %s Score: %.3f" %(best_cfg, best_score))

    dtc = DecisionTreeClassifier(max_depth = best_cfg[0], min_samples_split = best_cfg[1], random_state = 50).fit(np.concatenate(X_folds), np.concatenate(Y_folds))
    joblib.dump(dtc, wpath)

    plt.xticks(range(len(param_list)), param_list)
    plt.plot(range(len(param_list)), accuracy)
    plt.savefig(ppath)
    plt.show()

else:
    raise Exception("Invald Model name")
