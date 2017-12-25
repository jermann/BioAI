def randomWeights(self, w1, w2, w3, b): #*args
    weights = []
    for i in range(numInputs):
        weights[i] = random.uniform(-0.1, +0.1)


def setWeights(self): #*args
    self.weights = [] # deletes last list!
    r = self.numInputs - 1
    print r
    for i in range(r):
        d = i + 1
        temp = input("Weight %s: " % d)
        self.weights[i] = temp
    temp = input("Bias : ")
    self.weights[r] = temp

    print "pattern: %s, new weights: %s" % (pattern, ),
    for i in self.weights:
        print  str(i),
    print "\n"
