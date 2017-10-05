import numpy as np
import math

# make sure this class id compatable with sklearn's GaussianNB

class GaussianNB(object):
    def __init__(self ):
        # define all the model weights and state here
        pass

    def separateByClass(self, dataset, Y):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (Y[i] not in separated):
                separated[Y[i]] = []
            separated[Y[i]].append(vector)
        return separated

    def mean(self, numbers):
        return sum(numbers)/float(len(numbers))

    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

    def summarize(self, dataset):
        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]
        #del self.summaries[-1]
        return summaries

    def summarizeByClass(self, dataset, Y):
        separated = self.separateByClass(dataset, Y)
        summaries = {}
        for classValue, instances in separated.iteritems():
            summaries[classValue] = self.summarize(instances)
        return summaries

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    def calculateClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.iteritems():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
        return probabilities

    def predict_(self, summaries, inputVector):
        probabilities = self.calculateClassProbabilities(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def getPredictions(self, summaries, testSet):
        predictions = []
        for i in range(len(testSet)):
            result = self.predict_(summaries, testSet[i])
            predictions.append(result)
        return predictions

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i] == predictions[i]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0

    def fit(self, X , Y):
        self.summaries = self.summarizeByClass(X, Y)

    def predict(self, X ):
        predictions = self.getPredictions(self.summaries, X)
        return predictions
    # return a numpy array of predictions

    def score(self, X, Y):
        pred = self.predict(X)
        accuracy = self.getAccuracy(Y, predictions)
        return accuracy
