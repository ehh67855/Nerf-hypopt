import numpy as np
from optimization.optimizer import run_evolution

if __name__ == "__main__":
    # Run hyperparameter optimization
    best_hyperparams = run_evolution(n_generations=10, pop_size=20, surrogate_enabled=True)

    # Print the best found hyperparameters
    print("\nBest hyperparameters found:")
    for ind in best_hyperparams:
        print(f"Depth: {int(ind[0])}, Width: {int(ind[1])}, LR: {ind[2]:.6f}, N_rand: {int(ind[3])}")