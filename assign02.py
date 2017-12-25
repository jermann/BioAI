# Bio-Inspired Artificial Intelligence - Assignment 2 - Addendum
# Program: assign02-addendum.py - Robby the Robot
# Author: Alexander Jermann

from __future__ import division
import random, robby, Gnuplot

rw = robby.World(10, 10)
rw.graphicsOff()

g = None

def randomIndividual(length):
    x = ''
    for i in range(length):
        r = random.randint(0,6)
        x += str(r)
    return x

def makePopulation(size, length):
    return [randomIndividual(length) for i in range(0, size)]

def fitness(genome):
    totalReward = 0.0
    for i in range(0, 100):
        rw.goto(0, 0)
        rw.distributeCans(density = 0.50)
        reward = 0.0
        for i in range(0, 200):
            percept = rw.getPerceptCode()
            action = genome[percept]
            if action == '0':
                reward += rw.north()
            elif action == '1':
                reward += rw.south()
            elif action == '2':
                reward += rw.east()
            elif action == '3':
                reward += rw.west()
            elif action == '4':
                reward += rw.stay()
            elif action == '5':
                reward += rw.grab()
            elif action == '6':
                reward += rw.random()
        totalReward += reward
    avg = totalReward / 100.0
    return avg

def evaluateFitness(population):
    fitnesses = [fitness(genome) for genome in population]
    avgFit = sum(fitnesses) / len(population)
    bestFit = max(fitnesses)
    bestStrat = population[fitnesses.index(bestFit)]
    return fitnesses, avgFit, bestFit, bestStrat

def crossover(genome1, genome2):
    a = random.randrange(1,len(genome1))
    ngenome1 = genome1[:a] + genome2[a:]
    ngenome2 = genome2[:a] + genome1[a:]
    return (ngenome1, ngenome2)

def mutate(genome, mutationRate):
    ngenome = ''
    for i in range(0,len(genome)):
        allele = int(genome[i])
        if random.uniform(0,1) < mutationRate:
            ngenome += str(random.choice(range(1, allele) + range(allele + 1, 7)))
        else:
            ngenome += str(allele)
    return ngenome

def sortByFitness(population, fitnesses):
    tuples = [(fitnesses[i], population[i]) for i in range(0,len(population))]
    tuples.sort()
    sortedFitnessValues = [f for (f, g) in tuples]
    sortedGenomes = [g for (f, g) in tuples]
    return sortedGenomes, sortedFitnessValues

# Addendum - Took over function Weighted Choice from assign01.py
def weightedChoice(elements, weights):
    assert len(elements) == len(weights)
    assert all([v >= 0 for v in weights])
    total = sum(weights)
    r = random.uniform(0, total)
    w = 0
    for i in range(0, len(elements)):
        w += weights[i]
        if w > r:
            return elements[i]
    return random.choice(elements)

# Addendum - Changed Rank Selection function. Best Fitness grows faster but still gets stuck
def rankSelection(sortedGenomes, sortedFitnessValues):
    weights = [i for i in range(1, len(sortedGenomes)+1)]
    return weightedChoice(sortedGenomes, weights)

def selectPair(sortedGenomes, sortedFitnessValues):
    genome1 = rankSelection(sortedGenomes, sortedFitnessValues)
    genome2 = rankSelection(sortedGenomes, sortedFitnessValues)
    return genome1, genome2

def runGA(numberGenerations, populationSize = 200, crossoverRate = 1.0, mutationRate = 0.005, logFile="robby-default-24.txt", mode='normal'):
    assert populationSize % 2 == 0, "Population Size must be even"
    d = open(logFile, 'w')
    population = makePopulation(populationSize, 243)
    fitnesses, avgFit, bestFit, bestStrat = evaluateFitness(population)
    sortedGenomes, sortedFitnessValues = sortByFitness(population, fitnesses)
    for gen in range(1, numberGenerations + 1):
        # Build new population
        newPopulation = []
        for i in range(populationSize // 2):
            genome1, genome2 = selectPair(sortedGenomes, sortedFitnessValues)
            if random.uniform(0, 1) < crossoverRate:
                genome1, genome2 = crossover(genome1, genome2)
            genome1 = mutate(genome1, mutationRate)
            genome2 = mutate(genome2, mutationRate)
            newPopulation.append(genome1)
            newPopulation.append(genome2)
        population = newPopulation
        fitnesses, avgFit, bestFit, bestStrat = evaluateFitness(population)
        sortedGenomes, sortedFitnessValues = sortByFitness(population, fitnesses)
        if mode == 'normal' and gen % 10 == 0:
            output = report(gen, avgFit, bestFit, bestStrat, logFile)
            d.write(output)
            d.write("\n")
            d.flush()
            # rw.demo(bestStrat, steps = 200, init = 0.50)
        winner = 0.0
        if bestFit > winner:
            winner += bestFit
    d.close()
    return population[fitnesses.index(winner)]

def report(gen, avgFit, bestFit, bestStrat, logFile):
    output = "%3d avg: %9.7f best: %9.7f  %s" % (gen, avgFit, bestFit, bestStrat)
    plot(output)
    print output
    return output

def plot():
    global g
    g = Gnuplot.Gnuplot()
    command = "plot [0:500][0:400]"
    command += "'robby-default-24.txt' using 3 title 'average Fitness' with lines, "
    command += "'robby-default-24.txt' using 5 title 'best Fitness' with lines, "
    g(command)
    
plot()
