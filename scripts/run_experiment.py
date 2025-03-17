import numpy as np
import random
import json
import os
import multiprocessing
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from train import train_nerf  # Ensure this function is properly defined

# Hyperparameter search space
HYPERPARAM_SPACE = {
    "netdepth": (4, 12),
    "netwidth": (64, 512),
    "lrate": (1e-5, 1e-2),
    "N_rand": (128, 4096)
}

# Directory for logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def random_hyperparams():
    """Generate a random set of hyperparameters within bounds."""
    return {
        "netdepth": np.random.randint(*HYPERPARAM_SPACE["netdepth"]),
        "netwidth": np.random.randint(*HYPERPARAM_SPACE["netwidth"]),
        "lrate": 10 ** np.random.uniform(np.log10(HYPERPARAM_SPACE["lrate"][0]), np.log10(HYPERPARAM_SPACE["lrate"][1])),
        "N_rand": np.random.randint(*HYPERPARAM_SPACE["N_rand"])
    }


def train_surrogate(X_train, y_train):
    """Train a Gaussian Process surrogate model."""
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(X_train, y_train)
    return gp


def evaluate_individual(hyperparams, config_path):
    """Evaluate an individual by training the NeRF model."""
    try:
        result = train_nerf(hyperparams, config_path)
        return result["psnr"], result["training_time"], result["memory_usage"]
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return -np.inf, np.inf, np.inf  # Invalid results


def mutate(individual):
    """Mutate an individual by modifying a random hyperparameter."""
    param_to_mutate = random.choice(list(HYPERPARAM_SPACE.keys()))

    if param_to_mutate in ["netdepth", "netwidth", "N_rand"]:
        individual[param_to_mutate] += np.random.randint(-16, 16)
        individual[param_to_mutate] = max(HYPERPARAM_SPACE[param_to_mutate][0], min(individual[param_to_mutate], HYPERPARAM_SPACE[param_to_mutate][1]))
    elif param_to_mutate == "lrate":
        individual[param_to_mutate] *= 10 ** np.random.uniform(-0.1, 0.1)
        individual[param_to_mutate] = max(HYPERPARAM_SPACE["lrate"][0], min(individual[param_to_mutate], HYPERPARAM_SPACE["lrate"][1]))

    return individual,


def crossover(parent1, parent2):
    """Perform uniform crossover between two parents."""
    child = {key: random.choice([parent1[key], parent2[key]]) for key in HYPERPARAM_SPACE.keys()}
    return child,


def log_results(generation, population, fitnesses):
    """Log evolutionary results to a JSON file."""
    log_data = [{"generation": generation, "hyperparams": ind, "fitness": fit} for ind, fit in zip(population, fitnesses)]
    log_file = os.path.join(LOG_DIR, "evolution_log.json")

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.extend(log_data)

    with open(log_file, "w") as f:
        json.dump(data, f, indent=4)


def visualize_results():
    """Visualize PSNR evolution over generations."""
    log_file = os.path.join(LOG_DIR, "evolution_log.json")
    with open(log_file, "r") as f:
        data = json.load(f)

    generations = [entry["generation"] for entry in data]
    psnr_values = [entry["fitness"][0] for entry in data]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, psnr_values, marker='o', linestyle='-')
    plt.xlabel("Generation")
    plt.ylabel("PSNR")
    plt.title("PSNR Evolution Over Generations")
    plt.grid()
    plt.show()


def run_nsga2_optimization(config_path, generations=10, population_size=20):
    """Run NSGA-II optimization with DEAP."""
    
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))  # Maximize PSNR, minimize time and memory
    creator.create("Individual", dict, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, random_hyperparams)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, config_path=config_path)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=population_size)
    for gen in range(generations):
        print(f"\n===== Generation {gen} =====")

        # Evaluate individuals
        with multiprocessing.Pool(processes=min(4, population_size)) as pool:
            fitnesses = pool.map(toolbox.evaluate, population)

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Log results
        log_results(gen, population, fitnesses)

        # Apply selection, crossover, and mutation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        with multiprocessing.Pool(processes=min(4, len(invalid_ind))) as pool:
            fitnesses = pool.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    print("\nFinal Optimized NeRF Hyperparameters:", best_ind)

    with open(os.path.join(LOG_DIR, "best_hyperparams.json"), "w") as f:
        json.dump(best_ind, f, indent=4)

    return best_ind


if __name__ == "__main__":
    config_path = "../nerf-pytorch/configs/lego.txt"
    best_hyperparams = run_nsga2_optimization(config_path, generations=10, population_size=20)
    visualize_results()
