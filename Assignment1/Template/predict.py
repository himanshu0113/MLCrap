import os
import os.path
import argparse
import h5py

from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()


# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

X, Y_ = load_h5py(args.test_data)

Y = []
for row in Y_:
	pos = row.tolist().index(1)
	Y.append(pos)

dataset_name = args.test_data.split('/')
wpath = args.weights_path + dataset_name[-1] + '.pkl'

if(args.model_name == None):
	model = joblib.load(wpath)
elif args.model_name == 'GaussianNB':
	nwpath = args.weights_path + dataset_name[-1] + args.model_name + '.pkl'
	pass
elif args.model_name == 'LogisticRegression':
	pass
elif args.model_name == 'DecisionTreeClassifier':
	pass
else:
	raise Exception("Invald Model name")

pred = model.predict(X)
#print 'Actual \t Predicted'
#print str(x) + '\t' + str(y) for x, y in Y, pred

#saving results to file

with open(args.output_preds_file, "w") as text_file:
	for p in pred:
		text_file.write("{}\n".format(p))
