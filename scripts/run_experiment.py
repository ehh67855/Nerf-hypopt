import numpy as np
import random
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import multiprocessing
from train import train_nerf 

# Define hyperparameter search space
HYPERPARAM_SPACE = {
    "netdepth": (4, 12),  # Number of layers in MLP
    "netwidth": (64, 512),  # Number of neurons per layer
    "lrate": (1e-5, 1e-2),  # Learning rate
    "N_rand": (128, 4096)  # Batch size for training
}

# Generate a random hyperparameter set within bounds
def random_hyperparams():
    return {
        "netdepth": np.random.randint(HYPERPARAM_SPACE["netdepth"][0], HYPERPARAM_SPACE["netdepth"][1]),
        "netwidth": np.random.randint(HYPERPARAM_SPACE["netwidth"][0], HYPERPARAM_SPACE["netwidth"][1]),
        "lrate": 10 ** np.random.uniform(np.log10(HYPERPARAM_SPACE["lrate"][0]), np.log10(HYPERPARAM_SPACE["lrate"][1])),
        "N_rand": np.random.randint(HYPERPARAM_SPACE["N_rand"][0], HYPERPARAM_SPACE["N_rand"][1])
    }

# Train the surrogate model (Gaussian Process)
def train_surrogate(X_train, y_train):
    kernel = Matern(nu=2.5)  # Robust kernel for GP
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(X_train, y_train)
    return gp

# Mutate a hyperparameter set
def mutate(hyperparams):
    mutated = hyperparams.copy()
    param_to_mutate = random.choice(list(HYPERPARAM_SPACE.keys()))

    if param_to_mutate in ["netdepth", "netwidth", "N_rand"]:
        mutated[param_to_mutate] += np.random.randint(-16, 16)
        mutated[param_to_mutate] = max(HYPERPARAM_SPACE[param_to_mutate][0], min(mutated[param_to_mutate], HYPERPARAM_SPACE[param_to_mutate][1]))
    elif param_to_mutate == "lrate":
        mutated[param_to_mutate] *= 10 ** np.random.uniform(-0.1, 0.1)
        mutated[param_to_mutate] = max(HYPERPARAM_SPACE["lrate"][0], min(mutated[param_to_mutate], HYPERPARAM_SPACE["lrate"][1]))

    return mutated

# Crossover two parents
def crossover(parent1, parent2):
    child = {}
    for key in HYPERPARAM_SPACE.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

# Select top candidates using the surrogate
def select_best_candidates(population, surrogate_model, top_k=5):
    X_candidates = np.array([[p["netdepth"], p["netwidth"], p["lrate"], p["N_rand"]] for p in population])
    predicted_psnr = surrogate_model.predict(X_candidates)
    sorted_indices = np.argsort(predicted_psnr)[-top_k:]
    return [population[i] for i in sorted_indices]

# Evaluate NeRF training in parallel
def evaluate_population(population, config_path):
    results = []
    with multiprocessing.Pool(processes=min(4, len(population))) as pool:
        results = pool.starmap(train_nerf, [(p, config_path) for p in population])
    return results

# Main Surrogate-Assisted Evolutionary Algorithm
def surrogate_evolutionary_optimization(config_path, generations=10, population_size=20):
    # Step 1: Initialize population & evaluate some individuals
    population = [random_hyperparams() for _ in range(population_size)]
    results = evaluate_population(population, config_path)

    # Step 2: Convert results to NumPy format
    X_train = np.array([[p["netdepth"], p["netwidth"], p["lrate"], p["N_rand"]] for p in population])
    y_train_psnr = np.array([res["psnr"] for res in results])

    for gen in range(generations):
        print(f"\n===== Generation {gen+1} =====")

        # Step 3: Train Surrogate Model
        surrogate_model = train_surrogate(X_train, y_train_psnr)

        # Step 4: Select promising candidates
        best_candidates = select_best_candidates(population, surrogate_model, top_k=5)

        # Step 5: Generate new offspring using mutation & crossover
        new_population = []
        while len(new_population) < population_size:
            if random.random() < 0.5:
                new_population.append(mutate(random.choice(best_candidates)))
            else:
                p1, p2 = random.sample(best_candidates, 2)
                new_population.append(crossover(p1, p2))

        # Step 6: Evaluate NeRF training for selected candidates
        new_results = evaluate_population(new_population, config_path)

        # Step 7: Update training data
        new_X_train = np.array([[p["netdepth"], p["netwidth"], p["lrate"], p["N_rand"]] for p in new_population])
        new_y_train_psnr = np.array([res["psnr"] for res in new_results])

        X_train = np.vstack([X_train, new_X_train])
        y_train_psnr = np.hstack([y_train_psnr, new_y_train_psnr])

        population = new_population  # Update population

    # Step 8: Return best result found
    best_index = np.argmax(y_train_psnr)
    best_hyperparams = population[best_index]
    print(f"\nBest Found Hyperparameters: {best_hyperparams}")
    return best_hyperparams

# Run the optimizer
if __name__ == "__main__":
    config_path = "../nerf-pytorch/configs/lego.txt"
    best_hyperparams = surrogate_evolutionary_optimization(config_path, generations=10, population_size=20)
    print("\nFinal Optimized NeRF Hyperparameters:", best_hyperparams)
