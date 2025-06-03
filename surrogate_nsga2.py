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
NDIM = 5  # number of hyperparameters
# Define the bounds for each hyperparameter
BOUND_LOW = np.array([1e-4, 1, 16, 32, 100])  # learning_rate, layers, neurons, batch_size, decay_steps
BOUND_UP = np.array([1e-2, 12, 512, 2048, 1000])

# Define parameter names and types for consistent conversion
PARAM_SPECS = [
    ('learning_rate', float),
    ('num_hidden_layers', int),
    ('neurons_per_layer', int),
    ('batch_size', int),
    ('learning_decay_steps', int)
]

def array_to_dict(arr):
    """Convert numpy array to dictionary with proper types"""
    return {
        name: dtype(val) 
        for (name, dtype), val in zip(PARAM_SPECS, arr)
    }

def dict_to_array(d):
    """Convert dictionary to numpy array in consistent order"""
    return np.array([d[name] for name, _ in PARAM_SPECS])

# ========== DEAP Setup ==========
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))  # Maximize PSNR, minimize time/mem
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_bounded_float():
    return [random.uniform(low, up) for low, up in zip(BOUND_LOW, BOUND_UP)]

toolbox.register("individual", tools.initIterate, creator.Individual, create_bounded_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define custom mutation and crossover operators that work with arrays
def custom_mate(ind1, ind2):
    """Custom mate function that properly handles boundary constraints"""
    for i in range(len(ind1)):
        # Apply simulated binary crossover to each parameter individually
        if random.random() < 0.5:
            # Implement simulated binary crossover manually for a single parameter
            eta = 20.0
            x1, x2 = ind1[i], ind2[i]
            xl, xu = BOUND_LOW[i], BOUND_UP[i]
            
            # Check if the values are the same (avoid division by zero)
            if abs(x1 - x2) > 1e-10:
                if x1 > x2:
                    x1, x2 = x2, x1
                
                # Calculate beta
                rand = random.random()
                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** (-(eta + 1.0))
                
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                
                # Calculate child values
                c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
                
                # Ensure they are within bounds
                c1 = max(xl, min(xu, c1))
                c2 = max(xl, min(xu, c2))
                
                ind1[i], ind2[i] = c1, c2
            
    return ind1, ind2

def custom_mutate(individual, indpb=0.2):
    """Custom mutation that properly handles boundary constraints"""
    for i in range(len(individual)):
        if random.random() < indpb:
            # Apply polynomial mutation to each parameter individually
            eta = 20.0
            x = individual[i]
            xl = BOUND_LOW[i]
            xu = BOUND_UP[i]
            
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            
            # This polynomial mutation is based on the original DEAP implementation
            # but adapted to work with individual values instead of arrays
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)
            
            if rand < 0.5:
                xy = 1.0 - delta_1
                if xy > 0:  # Avoid issues with negative values
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    delta_q = 0.0
                individual[i] = x + delta_q * (xu - xl)
            else:
                xy = 1.0 - delta_2
                if xy > 0:  # Avoid issues with negative values
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - (val ** mut_pow)
                else:
                    delta_q = 0.0
                individual[i] = x + delta_q * (xu - xl)
            
            # Ensure the result is within bounds
            individual[i] = max(xl, min(xu, individual[i]))
    
    return individual,

# Register custom operators
toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

# ========== Logging ==========
log_dir = "logs/optimization/10k_iters"
os.makedirs(log_dir, exist_ok=True)

def log_generation(gen, pop, is_real_eval):
    results = []
    print(f"\n{'━'*20} GEN {gen} {'━'*20}")
    eval_types = {True: "Real", False: "Surrogate"}

    for i, (ind, is_real) in enumerate(zip(pop, is_real_eval)):
        params = array_to_dict(ind)
        fitness = ind.fitness.values
        results.append({
            "gen": gen,
            "index": i,
            "params": params,
            "fitness": list(fitness),
            "real_eval": is_real
        })

        print(f"{eval_types[is_real]} Eval | Ind #{i} | PSNR: {fitness[0]:.2f} | "
              f"Time: {fitness[1]:.2f}s | Mem: {fitness[2]:.2f}MB")
        print(f"           Params: lr={params['learning_rate']:.5f}, "
              f"layers={params['num_hidden_layers']}, "
              f"neurons={params['neurons_per_layer']}, "
              f"batch={params['batch_size']}, "
              f"decay={params['learning_decay_steps']}\n")

    # Dump results to JSON
    with open(os.path.join(log_dir, f"gen_{gen}.json"), "w") as f:
        json.dump(results, f, indent=2)

def evaluate_individual(individual):
    """Evaluate an individual using the proper parameter format"""
    params_dict = array_to_dict(individual)
    psnr, time_taken, mem, _ = evaluate_hyperparameters(params_dict)
    return psnr, time_taken, mem

def main(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    MU = 20  # population size
    NGEN = 40
    EVAL_EVERY = 5
    CXPB = 0.9

    surrogate = SurrogateModel()
    X_train = []
    y_train = {"psnr": [], "time": [], "memory": []}

    # Initialize population
    pop = toolbox.population(n=MU)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "min", "max"]

    # Evaluate initial population
    # ========== Initial Evaluation ==========
    for ind in pop:
        psnr, time_taken, mem = evaluate_individual(ind)
        ind.fitness.values = (psnr, time_taken, mem)
        X_train.append(list(ind))
        y_train["psnr"].append(psnr)
        y_train["time"].append(time_taken)
        y_train["memory"].append(mem)

    surrogate.fit(np.array(X_train), y_train)

    # NSGA-II selection to assign rank and crowding distance
    pop = toolbox.select(pop, len(pop))

    log_generation(0, pop, [True] * len(pop))

    for gen in range(1, NGEN + 1):
        offspring = []
        while len(offspring) < len(pop):
            parents = tools.selTournamentDCD(pop, 2)
            children = [toolbox.clone(ind) for ind in parents]

            if random.random() <= CXPB:
                toolbox.mate(children[0], children[1])
            for child in children:
                toolbox.mutate(child)
                del child.fitness.values
            offspring.extend(children)

        offspring = offspring[:len(pop)]  # Truncate to population size

        real_eval_indices = [i for i in range(len(offspring)) 
                            if gen % EVAL_EVERY == 0 or random.random() < 0.2]

        for i, ind in enumerate(offspring):
            if i in real_eval_indices:
                psnr, time_taken, mem = evaluate_individual(ind)
                ind.fitness.values = (psnr, time_taken, mem)
                X_train.append(list(ind))
                y_train["psnr"].append(psnr)
                y_train["time"].append(time_taken)
                y_train["memory"].append(mem)
            else:
                try:
                    pred = surrogate.predict(np.array([ind]))
                    ind.fitness.values = (pred[0][0], pred[1][0], pred[2][0])
                except NotFittedError:
                    ind.fitness.values = (0.0, 999.0, 999.0)

        if gen % EVAL_EVERY == 0:
            surrogate.fit(np.array(X_train), y_train)

        # NSGA-II selection: handles sorting + crowding distance
        pop = toolbox.select(pop + offspring, MU)

        # Track which individuals in the new population were real-evaluated
        offspring_ids = [id(ind) for ind in offspring]
        real_eval_ids = set(id(offspring[i]) for i in real_eval_indices)

        real_eval_flags = [id(ind) in real_eval_ids for ind in pop]

        log_generation(gen, pop, real_eval_flags)
        logbook.record(gen=gen, evals=len(real_eval_indices),
                    min=min(ind.fitness.values[0] for ind in pop),
                    max=max(ind.fitness.values[0] for ind in pop))
        print(logbook.stream)

    return pop, logbook

if __name__ == "__main__":
    pop, logbook = main()
