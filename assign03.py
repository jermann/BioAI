# Bio-Inspired Artificial Intelligence - Assignment 3
# Program: assign-03.py - Perceptrons
# Author: Alexander Jermann
# Template: Jim Marshall

import random

class Perceptron:

    def __init__(self):
        self.numInputs = 30 # Adjust this value to change the number of inputs
        self.weights = [random.uniform(-0.1, +0.1) for i in range(self.numInputs)]
        self.learningRate = 0.1

    # Function sets Weights as separate inputs. Is there a easier way to do it?
    def setWeights(self):
        self.weights = []
        r = self.numInputs - 1
        for i in range(r): # fot loop appends weights to empty list
            d = i + 1 # in order to display the correct number in the next line
            temp = input("Weight %s: " % d)
            self.weights.append(temp)
        temp = input("Bias : ")
        self.weights.append(temp) # appends bias outside loop, because single var.

    # Function prints list as string using a for loop.
    def showWeights(self):
        for i in self.weights:
            print str(i),

    def propagate(self, pattern):
        total = 0.0
        for i in range(len(pattern)):
            w = self.weights[i]
            p = pattern[i]
            total += w * p
        total += self.weights[self.numInputs - 1] # Again, bias added outside loop.
        if total >= 0:
            output = 1
        else:
            output = 0
        return output

    # Added a 'quiet' mode to the test function to prevent print outputs.
    def test(self, patterns, targets, mode = 'normal'):
        errorCounter = 0.0
        for pattern, target in zip(patterns, targets):
            output = self.propagate(pattern)
            if mode == 'normal' and output == target:
                print "%s --> %s" % (pattern, output)
            elif mode == 'normal':
                print "%s --> %s  (WRONG, should be %s)" % \
                    (pattern, output, target)
            elif output != target:
                errorCounter += 1 # Used later to calculate Percent Error
        errorTSS = self.totalError(patterns, targets)
        percError = 100.0 * (errorCounter / len(targets))
        print "TSS error = %.1f" % errorTSS
        return errorTSS, percError # Returning two different error measures

    def adjustWeights(self, pattern, target, mode='normal'):
        output = self.propagate(pattern)
        error = target - output
        weightLength = self.numInputs - 1 #needed?
        if output == target:
            pass
        else:
            for i in range(len(pattern)):
                x = pattern[i]
                weightChange = self.learningRate * error * x
                self.weights[i] += weightChange
            biasChange = self.learningRate * error
            self.weights[weightLength] += biasChange
            if mode == 'normal':
                print "pattern: %s, new weights: %s" % (pattern, self.weights)

    def totalError(self, patterns, targets):
        squaredError = 0
        hello = zip(patterns,targets)
        print hello
        for pattern, target in zip(patterns, targets):
            output = self.propagate(pattern)
            squaredError += (target - output) ** 2
        return squaredError

    def train(self, patterns, targets):
        epochCounter = 0.0
        error = self.totalError(patterns, targets)
        while error > 0:
            # one training epoch
            for pattern, target in zip(patterns, targets):
                # Calling adjustWeights with 'quiet' mode.
                self.adjustWeights(pattern, target, mode = 'quiet')
            epochCounter += 1 # Used to calculate Epochs
            error = self.totalError(patterns, targets)
        print "all patterns learned after %i epochs" % epochCounter
        return epochCounter

    # This Function takes a patterns & a targets dataset, and a number of times
    # the experiment should be repeated. It was used to solve exercise 7.
    def experiment(self, patternsFilename, targetsFilename, iterations):
        totalEpochs = 0.0
        totalError = 0.0
        totalPercError = 0.0
        cpats, ctargs = readDataset(patternsFilename, targetsFilename)
        cpats1 = cpats[:len(cpats)/2]
        cpats2 = cpats[len(cpats)/2:]
        ctargs1 = ctargs[:len(ctargs)/2]
        ctargs2 = ctargs[len(ctargs)/2:]
        for i in range(0, iterations):
            epochCounter = self.train(cpats1, ctargs1)
            totalEpochs += epochCounter
            errorTSS, percError = self.test(cpats2,ctargs2, mode = 'quiet')
            totalError += errorTSS
            totalPercError += percError
            self.weights = [random.uniform(-0.1, +0.1) for i in range(self.numInputs)]
        avgEpochs = totalEpochs / iterations
        avgError = totalError / iterations
        avgPercError = totalPercError / iterations
        print "\nNumber of Iterations of Experiment: %i" % iterations
        print "Average Epochs to train Perceptron: %i" % avgEpochs
        print "Average TSS error: %.2f" % avgError
        print "Average Error of Perceptron on new Data after training: %.2f%%" % avgPercError

#------------------------------------------------------------------
# some input and target patterns

patterns2 = [[0,0], [0,1], [1,0], [1,1]]

ANDtargets  = [0, 0, 0, 1]
ORtargets   = [0, 1, 1, 1]
NANDtargets = [1, 1, 1, 0]
NORtargets  = [1, 0, 0, 0]
XORtargets  = [0, 1, 1, 0]  # not linearly separable

TWOtargets  = [1, 0, 1, 1]
NTWOtargets = [0, 1, 0, 0]

patterns3 = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]

CARRYtargets = [0, 0, 0, 1, 0, 1, 1, 1]
SUMtargets   = [0, 1, 1, 0, 1, 0, 0, 1]

#------------------------------------------------------------------
# for part 7

def readDataset(patternsFilename, targetsFilename):
    file = open(patternsFilename)
    patterns = [[eval(x) for x in line.split()] for line in file]
    file.close()
    file = open(targetsFilename)
    targets = [eval(line) for line in file]
    file.close()
    return patterns, targets

# usage:
# cpats, ctargs = readDataset('cancerPatterns.txt', 'cancerTargets.txt')
