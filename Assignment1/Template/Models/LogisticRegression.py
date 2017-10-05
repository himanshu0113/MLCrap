import numpy as np


# make sure this class id compatable with sklearn's LogisticRegression

class LogisticRegression(object):
    learning_rate = 5e-5

    def __init__(self, penalty='l2' , C=1.0 , max_iter=100 , verbose=0):
        self.lamda = (1.0/C)
        self.iter = max_iter

    def sigmoid(self, scores):
        return 1.0 / (1 + np.exp(-scores))

    def fit(self, X , Y):
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        self.weights = np.zeros(X.shape[1])
        m = len(X)

        for step in xrange(self.iter):
            scores = np.dot(X, self.weights)
            predictions = self.sigmoid(scores)
            #print 'predictions' + str(predictions)
            # Update weights with gradient
            output_error_signal = Y - predictions
            #print 'output_error_signal' + str(output_error_signal)
            gradient_partial = np.multiply(np.dot(X.T, output_error_signal), (1.0/m))
            #print 'grad par' + str(gradient_partial)
            gradient_reg = np.multiply((self.lamda/m), np.multiply(self.weights, self.weights))
            #print 'grad reg' + str(gradient_reg)
            #gradient_reg[0] = 0
            gradient = np.add(gradient_partial, gradient_reg)
            #gradient = np.dot(X.T, output_error_signal)
            self.weights += np.multiply(LogisticRegression.learning_rate, gradient)
            #self.weights += LogisticRegression.learning_rate * gradient
            #print self.weights


    def predict(self, X ):
        X = np.array(X)
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))

        final_scores = np.dot(X, self.weights)
        preds = np.round(self.sigmoid(final_scores))
        return preds.astype(int)+1
        # return a numpy array of predictions



    def score(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        pred = predict(X)
        res = (preds == Y).sum().astype(float) / len(preds)
        print res
        return res
