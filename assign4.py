# Bio-Inspired Artificial Intelligence - Assignment 4
# Program: assign-04.py - The Backpropagation Learning Algorithm
# Author: Alexander Jermann
# Template: Jim Marshall

from __future__ import division
import random, math
import NNinspector

class NeuralNetwork:

    def __init__(self):
        # unit activations
        self.output = 0
        self.hidden1 = 0
        self.hidden2 = 0
        # output unit weights
        self.output_bias = random.uniform(-0.1, +0.1)
        self.output_w1 = random.uniform(-0.1, +0.1)
        self.output_w2 = random.uniform(-0.1, +0.1)
        # hidden1 unit weights
        self.hidden1_bias = random.uniform(-0.1, +0.1)
        self.hidden1_w1 = random.uniform(-0.1, +0.1)
        self.hidden1_w2 = random.uniform(-0.1, +0.1)
        # hidden2 unit weights
        self.hidden2_bias = random.uniform(-0.1, +0.1)
        self.hidden2_w1 = random.uniform(-0.1, +0.1)
        self.hidden2_w2 = random.uniform(-0.1, +0.1)
        # learning parameters
        self.tolerance = 0.1
        self.learningRate = 0.1 # 0.5
        self.momentum = 0.9
        self.sigmoidPrimeOffset = 0  # 0.1
        # support for momentum
        self.output_bias_change = 0
        self.hidden1_bias_change = 0
        self.hidden2_bias_change = 0
        self.output_w1_change = 0
        self.output_w2_change = 0
        self.hidden1_w1_change = 0
        self.hidden1_w2_change = 0
        self.hidden2_w1_change = 0
        self.hidden2_w2_change = 0

    def __str__(self):
        s = "           bias     w1      w2\n"
        s += "output   %+.4f %+.4f %+.4f\n" % \
             (self.output_bias, self.output_w1, self.output_w2)
        s += "hidden1  %+.4f %+.4f %+.4f\n" % \
             (self.hidden1_bias, self.hidden1_w1, self.hidden1_w2)
        s += "hidden2  %+.4f %+.4f %+.4f" % \
             (self.hidden2_bias, self.hidden2_w1, self.hidden2_w2)
        return s

    def setWeights(self, weightList):
        assert type(weightList) == list and len(weightList) == 9, \
            "a list of 9 weight/bias values is required"
        # weights in weightList must be given in the order below:
        self.output_bias  = weightList[0]
        self.output_w1    = weightList[1]
        self.output_w2    = weightList[2]
        self.hidden1_bias = weightList[3]
        self.hidden1_w1   = weightList[4]
        self.hidden1_w2   = weightList[5]
        self.hidden2_bias = weightList[6]
        self.hidden2_w1   = weightList[7]
        self.hidden2_w2   = weightList[8]

    def initialize(self):
        self.setWeights([random.uniform(-0.1, +0.1) for i in range(9)])
        print "weights randomized"

    def test(self, patterns, targets):
        for pattern, target in zip(patterns, targets):
            output = self.propagate(pattern)
            if abs(target - output) <= self.tolerance:
                print "%s --> %s" % (pattern, output)
            else:
                print "%s --> %s \t(WRONG, should be %s)" % (pattern, output, target)
        error, correct = self.totalError(patterns, targets)
        print "TSS error = %.5f, correct = %.3f" % (error, correct)

    #--------------------------------------------------------------------------

    # returns the output produced by the network for the given pattern
    def propagate(self, pattern):
        x1, x2 = pattern
        self.hidden1 = self.sigmoid(x1 * self.hidden1_w1 + x2 * self.hidden1_w2 + self.hidden1_bias)
        self.hidden2 = self.sigmoid(x1 * self.hidden2_w1 + x2 * self.hidden2_w2 + self.hidden2_bias)
        self.output =  self.sigmoid(self.hidden1 * self.output_w1 + self.hidden2 * self.output_w2 + self.output_bias)
        return self.output

    def sigmoid(self, a):
         return 1.0 / (1.0 + math.exp(-1.0 * a))

    # Returns a tuple of values (e, c) where e is the total sum squared error
    # for all patterns in the given dataset, and c is the fraction of output
    # values that are within self.tolerance of the given target values
    def totalError(self, patterns, targets):
        c, e = 0, []
        for pattern, target in zip(patterns, targets):
            e.append((target - self.propagate(pattern))**2) # Adjust so not changed twice
            if abs(target - self.propagate(pattern)) <= self.tolerance:
                c += 1
        tss = sum(e)
        c = c / len(patterns)
        return (tss, c)

    # updates network weights and biases for the given pattern and target
    def adjustWeights(self, pattern, target):
        delta_output = 0.0
        x1, x2 = pattern
        activation = self.propagate(pattern)

        # Compute Delta values
        delta_output = (target - activation) * activation * (1.0 - activation)
        delta_h1 = delta_output * self.output_w1 * self.hidden1 * (1.0 - self.hidden1) # Already use adjusted weights?
        delta_h2 = delta_output * self.output_w2 * self.hidden2 * (1.0 - self.hidden2)

        # Compute weight change from hidden to output
        self.output_w1_change   = self.learningRate * delta_output * self.hidden1 + self.momentum * self.output_w1_change
        self.output_w2_change   = self.learningRate * delta_output * self.hidden2 + self.momentum * self.output_w2_change
        self.output_bias_change = self.learningRate * delta_output + self.momentum * self.output_bias_change

        # Update output weights and biases
        self.output_w1   += self.output_w1_change
        self.output_w2   += self.output_w2_change
        self.output_bias += self.output_bias_change

        # Compute weight change from input to hidden
        self.hidden1_w1_change     = self.learningRate * delta_h1 * x1 + self.momentum * self.hidden1_w1_change
        self.hidden1_w2_change     = self.learningRate * delta_h1 * x2 + self.momentum * self.hidden1_w2_change
        self.hidden1_bias_change   = self.learningRate * delta_h1      + self.momentum * self.hidden1_bias_change
        self.hidden2_w1_change     = self.learningRate * delta_h2 * x1 + self.momentum * self.hidden2_w1_change
        self.hidden2_w2_change     = self.learningRate * delta_h2 * x2 + self.momentum * self.hidden2_w2_change
        self.hidden2_bias_change   = self.learningRate * delta_h2      + self.momentum * self.hidden2_bias_change

        # Update input to hidden weights and biases
        self.hidden1_w1     += self.hidden1_w1_change
        self.hidden1_w2     += self.hidden1_w2_change
        self.hidden1_bias   += self.hidden1_bias_change
        self.hidden2_w1     += self.hidden2_w1_change
        self.hidden2_w2     += self.hidden2_w2_change
        self.hidden2_bias   += self.hidden2_bias_change

    # Trains the network on all of the given patterns and targets
    # until all outputs are within tolerance of the targets
    def train(self, patterns, targets, mode='normal'):
        tss, fraction  = self.totalError(patterns, targets)
        counter = 0
        while fraction != 1:
            # one training epoch
            for pattern, target in zip(patterns, targets):
                self.adjustWeights(pattern, target)
            counter += 1
            tss, fraction = self.totalError(patterns, targets)
            if mode == 'normal':
                print "Epoch # %4i: TSS error = %.5f, correct = %.3f" % \
                                                (counter, tss, fraction)
        return counter

    def experiment(self, patterns, targets):
        total = 0.0
        self.initialize()
        for i in range(5):
            total += self.train(patterns, targets, mode='quiet')
            self.initialize()
        return total / 5.0

#------------------------------------------------------------------------------
# input and target patterns

inputs = [[0,0], [0,1], [1,0], [1,1]]

ANDtargets  = [0, 0, 0, 1]
ORtargets   = [0, 1, 1, 1]
NANDtargets = [1, 1, 1, 0]
NORtargets  = [1, 0, 0, 0]
XORtargets  = [0, 1, 1, 0]  # not linearly separable
XNORtargets = [1, 0, 0, 1]  # not linearly separable

n = NeuralNetwork()
