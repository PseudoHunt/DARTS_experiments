import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from deap import base, creator, tools, algorithms

# Define the CNN model with a Pointwise Convolution (1x1 Convolution) layer
class PointwiseCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PointwiseCNN, self).__init__()
        # Define the Pointwise convolution layer (1x1 Conv)
        self.pointwise_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.fc = nn.Linear(output_channels, 10)  # Assuming 10 output classes for a classification task

    def forward(self, x):
        x = self.pointwise_conv(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define an evaluation function that computes the loss for the Pointwise CNN model
def evaluate(individual):
    model = PointwiseCNN(input_channels=3, output_channels=individual.shape[0])
    with torch.no_grad():
        # Assign the individual's weights to the model
        model.pointwise_conv.weight.data = torch.tensor(individual, dtype=torch.float32).view_as(model.pointwise_conv.weight)
        model.pointwise_conv.bias.data.fill_(0)  # Optionally initialize bias to zero

    # Define a dummy input and target for the example (use your actual dataset)
    input_data = torch.randn(1, 3, 32, 32)  # Example: 1 image of size 32x32 with 3 channels
    target = torch.tensor([0])  # Example: Target label

    # Forward pass
    output = model(input_data)
    loss = F.cross_entropy(output, target)  # Cross-entropy loss for classification
    return loss.item(),  # Return the loss (must be a tuple)

# Genetic algorithm setup using DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize loss
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)  # Initialize weights randomly between -1.0 and 1.0
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3*3)  # Individual has 3x3 weights for 1x1 Conv
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Gaussian mutation
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Main function to run the evolutionary algorithm
def main():
    population = toolbox.population(n=50)  # Initialize population
    hof = tools.HallOfFame(1)  # Hall of Fame to store the best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the evolutionary algorithm
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                                              stats=stats, halloffame=hof, verbose=True)

    # Print the best individual
    print("Best individual:", hof[0])
    return population, logbook, hof

if __name__ == "__main__":
    main()
