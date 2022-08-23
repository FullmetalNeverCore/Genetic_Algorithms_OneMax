import random
import matplotlib.pyplot as plt

onemaxlen = 100    


POPULATION_SIZE = 200  
P_CROSSOVER = 0.9      
P_MUTATION = 0.1      
maxgen = 50    

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class BeautifulGraph(object):
    def __init__(self,maxFitValues,meanFitValues):
        plt.plot(maxFitValues, color='red')
        plt.plot(meanFitValues, color='green')
        plt.xlabel('Gen')
        plt.ylabel('Max. Adapted')
        plt.title('Medium:')
        plt.show()

class FitnessMax():
    def __init__(self):
        self.values = [0]


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


def oneMaxFitness(individual):
    return sum(individual),

def individualCreator():
    return Individual([random.randint(0, 1) for i in range(onemaxlen)])

def populationCreator(n = 0):
    return list([individualCreator() for i in range(n)])


population = populationCreator(n=POPULATION_SIZE)
generationCounter = 0

fitnessValues = list(map(oneMaxFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitValues = []
meanFitValues = []

def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind

def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring

def cxOnePoint(child1, child2):
    s = random.randint(2, len(child1)-3)
    child1[s:], child2[s:] = child2[s:], child1[s:]

def mutFlipBit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1


fitnessValues = [individual.fitness.values[0] for individual in population]

#Main Cycle 

while max(fitnessValues) < onemaxlen and generationCounter < maxgen:
    generationCounter += 1
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION: 
            mutFlipBit(mutant, indpb=1.0/onemaxlen)

    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.fitness.values[0] for ind in population]


    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitValues.append(maxFitness)
    meanFitValues.append(meanFitness)
    print(f'Gen: {generationCounter}: Max. Adapted: {maxFitness}, Medium: {meanFitness}')    

    best_index = fitnessValues.index(max(fitnessValues))
    print("The best one:",*population[best_index],"\n")

BeautifulGraph(maxFitValues,meanFitValues)