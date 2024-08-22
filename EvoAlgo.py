#!pip install deap

import random
from deap import base, creator, tools, algorithms

# Define the fitness function
def fitness_function(individual):
    # Example: Maximize the number of 1s in the gene
    return sum(individual),

# Setup DEAP framework
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 12)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operators
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    population = toolbox.population(n=100)
    generations = 50
    crossover_prob = 0.7
    mutation_prob = 0.2

    # Run the algorithm
    algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations,
                        stats=None, halloffame=None, verbose=True)

    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]
    print(f"Optimized Gene: {best_individual}, Fitness: {best_individual.fitness.values[0]}")

if __name__ == "__main__":
    main()

