import random
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# Import data from data file
file = open("auto-mpg.data")
items = {}
indx = 0
for line in file:
    attrs = line.split(None, 8)
    # mpg, cylinders, displacement, horsepower, weight, acceleration, year, car name
    items[indx] = (float(attrs[0]), int(attrs[1]), float(attrs[2]), float(attrs[3]), float(attrs[4]), float(attrs[5]),
                   int("19" + attrs[6]), attrs[8][1:-2])
    indx += 1
file.close()

creator.create("Fitness", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_item", random.randint, 0, len(items) - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate_fitness(individual):
    """
    This function calculates the fitness score of an individual. The fitness score is simply a tuple containing the mpg,
    cylinders, displacement, horsepower, weight, acceleration, and year of the car.

    :param individual: the current individual
    :return: a tuple containing the relevant values
    """
    (i,) = individual
    mpg = items[i][0]
    cylinders = items[i][1]
    displacement = items[i][2]
    horsepower = items[i][3]
    weight = items[i][4]
    acceleration = items[i][5]
    year = items[i][6]
    return mpg, cylinders, displacement, horsepower, weight, acceleration, year


def cx_set(ind1, ind2):
    """
    This function returns two children made from the two individuals passed in as arguments. The children are found by
    the two items in the entire collection that have the lowest squared difference between the average of the parents
    and the children. Rather than creating new individuals, the parents' data are modified to represent the children's.

    :param ind1: the first parent
    :param ind2: the second parent
    :return: a tuple containing two individuals representing the children
    """
    (index1,) = ind1
    (index2,) = ind2
    car1 = items[index1]
    car2 = items[index2]
    avg_mpg = (car1[0] + car2[0]) / 2
    avg_cylinders = (car1[1] + car2[1]) / 2
    avg_displacement = (car1[2] + car2[2]) / 2
    avg_horsepower = (car1[3] + car2[3]) / 2
    avg_weight = (car1[4] + car2[4]) / 2
    avg_acceleration = (car1[5] + car2[5]) / 2
    avg_year = (car1[6] + car2[6]) / 2
    fitnesses = [evaluate_fitness([i]) for i in range(len(items))]
    min_indices = [0, 0]
    min_diffs = [float("inf"), float("inf")]
    for i in range(len(fitnesses)):
        mpg = fitnesses[i][0]
        cylinders = fitnesses[i][1]
        displacement = fitnesses[i][2]
        horsepower = fitnesses[i][3]
        weight = fitnesses[i][4]
        acceleration = fitnesses[i][5]
        year = fitnesses[i][6]
        diff = (mpg - avg_mpg) ** 2 + (cylinders - avg_cylinders) ** 2 + (displacement - avg_displacement) ** 2 + \
               (horsepower - avg_horsepower) ** 2 + (weight - avg_weight) ** 2 + \
               (acceleration - avg_acceleration) ** 2 + (year - avg_year) ** 2
        if diff < min_diffs[0]:
            min_diffs[0] = diff
            min_indices[0] = i
        elif diff < min_diffs[1]:
            min_diffs[1] = diff
            min_indices[1] = i
    ind1.pop()
    ind1.add(min_indices[0])
    ind2.pop()
    ind2.add(min_indices[1])
    return ind1, ind2


def mutate(individual):
    """
    This function introduces mutations to an individual. If an individual is mutated, it is simply replaced with a
    random item from the entire collection. Rather than creating a new individual, this individual's data is replaced
    with the new item's data.

    :param individual: the current individual
    :return:
    """
    if random.random() < 0.5:
        individual.pop()
        individual.add(random.randint(0, len(items) - 1))
    return individual,


def main(population_size, max_generations):
    """
    This function is the main genetic algorithm. It generates various elements until the target phrase is reached.
    Every generation, it prints relevant information to the terminal. Once it terminates, it will have found a Pareto
    optimal set of items.

    :param population_size: the size of the population
    :param max_generations: the number of generations before terminating
    :return the population, statistics, and the best individuals
    """
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", cx_set)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)

    NGEN = max_generations
    MU = population_size
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)

    return pop, stats, hof


def get_car_description(car):
    """
    This function returns a string representing the description of a car.

    :param car: a tuple containing all the information representing a car.
    :return: a string describing the car
    """
    mpg = car[0]
    cylinders = car[1]
    displacement = car[2]
    horsepower = car[3]
    weight = car[4]
    acceleration = car[5]
    year = car[6]
    name = car[7]
    return "{0} {1}, MPG: {2}, Cylinders: {3}, Displacement: {4}, Horsepower: {5}, " \
           "Weight: {6}, Acceleration: {7}".format(year, name, mpg, cylinders, displacement, horsepower, weight,
                                                   acceleration)


"""
This executes the genetic algorithm with the following parameters:
- Population size: 50
- Max Generations: 200
These values can be modified. After the genetic algorithm terminates, we print every Pareto optimal individual.
"""
if __name__ == '__main__':
    population, statistics, best = main(50, 200)
    pareto_set = set()
    for ind in best:
        (index,) = ind
        pareto_set.add(items[index])
    print(len(pareto_set))
    for item in pareto_set:
        print(get_car_description(item))
