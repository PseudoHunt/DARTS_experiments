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

    # Statistics to keep track of
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", max)

    # Record the hall of fame
    halloffame = tools.HallOfFame(1)

    for gen in range(generations):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the population with the offspring
        population[:] = offspring

        # Update the hall of fame with the best individuals
        halloffame.update(population)

        # Print the best individual in the current generation
        best_individual = halloffame[0]
        print(f"Generation {gen + 1}: Best Gene = {best_individual}, Fitness = {best_individual.fitness.values[0]}")

    # Final output: the best individual found
    best_individual = halloffame[0]
    print(f"Optimized Gene: {best_individual}, Fitness: {best_individual.fitness.values[0]}")

if __name__ == "__main__":
    main()
