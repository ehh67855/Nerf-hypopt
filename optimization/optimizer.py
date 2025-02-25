import numpy as np
import random
from deap import base, creator, tools
from optimization.surrogate import SurrogateModel
from scripts.train import train_nerf
from optimization.search_space import BOUNDS

# Define fitness function (maximize PSNR, minimize time & memory)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Search space bounds
HYPERPARAMETER_BOUNDS = {
    "netdepth": (4, 12),
    "netwidth": (64, 512),
    "lrate": (1e-5, 1e-2),
    "N_rand": (128, 1024)
}
BOUNDS = np.array([HYPERPARAMETER_BOUNDS[key] for key in HYPERPARAMETER_BOUNDS.keys()])

# Initialize individuals
toolbox.register("attr_float", lambda low, high: random.uniform(low, high))
toolbox.register("individual", tools.initCycle, creator.Individual,
                 [lambda: toolbox.attr_float(low, high) for low, high in BOUNDS], n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS[:, 0], high=BOUNDS[:, 1], eta=0.5, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def run_evolution(n_generations=10, pop_size=20, surrogate_enabled=True):
    """ Runs NSGA-II with Gaussian Process surrogate model. """
    pop = toolbox.population(n=pop_size)
    surrogate = SurrogateModel()
    
    history_X, history_Y = [], []  # Store past evaluations

    for gen in range(n_generations):
        print(f"\nGeneration {gen+1}/{n_generations}")

        if surrogate_enabled and gen > 0:
            # Use surrogate to select promising candidates
            candidates = np.array([list(ind) for ind in pop])
            next_suggestion = surrogate.suggest_next(candidates)
            print(f"Surrogate suggested: {next_suggestion}")

            # Train NeRF on selected candidate
            psnr, train_time, memory_usage = train_nerf(next_suggestion)  # Remove config_path parameter
            fitness = (psnr, train_time, memory_usage)

            # Update individual fitness
            for ind in pop:
                if list(ind) == list(next_suggestion):
                    ind.fitness.values = fitness
                    break

            history_X.append(next_suggestion)
            history_Y.append(fitness)

            # Train surrogate on new data
            surrogate.fit(np.array(history_X), np.array(history_Y))

        else:
            # Full evaluation (expensive)
            for ind in pop:
                hyperparams = list(ind)
                psnr, train_time, memory_usage = train_nerf(hyperparams)  # Remove config_path parameter
                ind.fitness.values = (psnr, train_time, memory_usage)

                # Store data
                history_X.append(hyperparams)
                history_Y.append(ind.fitness.values)

            # Train surrogate on initial data
            surrogate.fit(np.array(history_X), np.array(history_Y))

        # Apply genetic operations
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop[:] = offspring

    return pop
