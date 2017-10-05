
# coding: utf-8

# In[1]:

import json
from sklearn import svm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load the test data
def load_json(filename):
    X = []
    Y = []
    with open(filename) as data_file:
        data = json.load(data_file)
        for line in data:
            X.append(line["X"])
            Y.append(line["Y"])
    #print("keys: %s" % hf.keys())
    return X,Y

def load_json_test(filename):
    X = []
    with open(filename) as data_file:
        data = json.load(data_file)
        for line in data:
            X.append(line["X"])
    #print("keys: %s" % hf.keys())
    return X

X,Y = load_json('train.json')
X_test= load_json_test('test.json')

print 'done'


# In[2]:

Z= [list(a) for a in zip(X,Y)]
#z_new = set(z)

#removing redundancy
Z_new = []
for z in Z:
    if z not in Z_new:    
        Z_new.append(z)

print 'done'


# In[ ]:

from random import randint
X_train,Y_train = zip(*Z_new)
X_train = list(X_train)
Y_train = list(Y_train)

leny = len(Y_train)
i = 0

while i<leny:
    if Y_train[i] == 0:
        del X_train[i]
        del Y_train[i]
        leny= leny-1
    elif Y_train[i] == 5 and randint(0,4) in [1,2,3,4]:
        del X_train[i]
        del Y_train[i]
        leny= leny-1
    elif Y_train[i] == 1 and randint(0,1) ==1:
        del X_train[i]
        del Y_train[i]
        leny= leny-1
    i=i+1

print 'done'


# In[ ]:

X_str = []
for x in X_train:
    X_str.append(' '.join(str(e) for e in x))

print X_str[:10]


# In[ ]:

print X_str[:10]


# In[ ]:

vectorizor = CountVectorizer(analyzer='word', token_pattern='\d+')
X_train = vectorizor.fit_transform(X_str)

print X_train.shape[0]


# In[ ]:

Y_train = np.array(Y_train).flatten()
classifier = svm.LinearSVC()
print 'done'
classifier.fit(X_train, Y_train)
print 'done'


# In[ ]:

test_str = []
for x in X_test:
    test_str.append(' '.join(str(e) for e in x))

print test_str[:2]

#vectorizor2 = CountVectorizer(analyzer='word', token_pattern='\d+')
test = vectorizor.transform(test_str)

print test[1]

predicted = classifier.predict(test)
print 'done'


# In[ ]:

import csv

with open('result.csv', 'wb') as myfile:
    wr = csv.writer(myfile, delimiter = ',')
    for i in range(1,len(predicted)+1):
        wr.writerow([i, predicted[i-1]])

print 'done'

