import array
import random
import numpy as np
import json
import os
import time as pytime
from deap import base, creator, tools, algorithms
from SurrogateModel import SurrogateModel 
from run_nerf import evaluate_hyperparameters 
from sklearn.exceptions import NotFittedError

# ========== Config ==========
NDIM = 6  # number of hyperparameters
BOUND_LOW = [0.0001, 1, 16, 32, 1e-5, 1]  # example: learning rate, layers, etc.
BOUND_UP  = [0.01, 6, 256, 512, 1e-1, 10]

# ========== DEAP Setup ==========
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))  # Maximize PSNR, minimize time/mem
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: [random.uniform(lo, hi) for lo, hi in zip(BOUND_LOW, BOUND_UP)])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
toolbox.register("select", tools.selNSGA2)

# ========== Logging ==========
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def log_generation(gen, pop, is_real_eval):
    results = [
        {
            "gen": gen,
            "params": list(ind),
            "fitness": list(ind.fitness.values),
            "real_eval": is_real_eval[i]
        }
        for i, ind in enumerate(pop)
    ]
    with open(os.path.join(log_dir, f"gen_{gen}.json"), "w") as f:
        json.dump(results, f, indent=2)

# ========== Main Optimization Loop ==========
def main(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    MU = 20
    NGEN = 40
    EVAL_EVERY = 5  # retrain surrogate every N generations
    CXPB = 0.9

    surrogate = SurrogateModel()
    X_train, y_train = [], {"psnr": [], "time": [], "memory": []}

    pop = toolbox.population(n=MU)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "min", "max"]

    # Evaluate entire initial population with true function
    for ind in pop:
        psnr, time_taken, mem, _ = evaluate_hyperparameters(ind)
        ind.fitness.values = (psnr, time_taken, mem)
        X_train.append(list(ind))
        y_train["psnr"].append(psnr)
        y_train["time"].append(time_taken)
        y_train["memory"].append(mem)

    # Train surrogate model
    surrogate.fit(np.array(X_train), y_train)
    log_generation(0, pop, [True] * len(pop))
    logbook.record(gen=0, evals=len(pop), min=min(ind.fitness.values[0] for ind in pop),
                   max=max(ind.fitness.values[0] for ind in pop))

    for gen in range(1, NGEN + 1):
        # Generate offspring
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Determine which individuals to evaluate with the real function
        real_eval_indices = [i for i in range(len(offspring)) if gen % EVAL_EVERY == 0 or random.random() < 0.2]
        for i, ind in enumerate(offspring):
            if i in real_eval_indices:
                psnr, time_taken, mem, _ = evaluate_hyperparameters(ind)
                ind.fitness.values = (psnr, time_taken, mem)

                X_train.append(list(ind))
                y_train["psnr"].append(psnr)
                y_train["time"].append(time_taken)
                y_train["memory"].append(mem)
            else:
                try:
                    pred = surrogate.predict(np.array([ind]))  # [[...]] shape
                    ind.fitness.values = (pred[0][0], pred[1][0], pred[2][0])
                except NotFittedError:
                    ind.fitness.values = (0.0, 999.0, 999.0)  # fallback if surrogate fails

        # Retrain surrogate every EVAL_EVERY generations
        if gen % EVAL_EVERY == 0:
            surrogate.fit(np.array(X_train), y_train)

        pop = toolbox.select(pop + offspring, MU)
        log_generation(gen, pop, [(i in real_eval_indices) for i in range(len(pop))])
        logbook.record(gen=gen, evals=len(real_eval_indices),
                       min=min(ind.fitness.values[0] for ind in pop),
                       max=max(ind.fitness.values[0] for ind in pop))
        print(logbook.stream)

    return pop, logbook

if __name__ == "__main__":
    pop, logbook = main()
 