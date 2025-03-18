import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
from run_nerf import evaluate_hyperparameters
import torch
import psutil

# Create fitness class for multi-objective optimization
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0, -1.0))  # PSNR, Time, CPU, Memory
creator.create("Individual", list, fitness=creator.FitnessMulti)

class SurrogateModel:
    def __init__(self):
        # Initialize separate GP models for each objective
        self.gp_psnr = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(),
            n_restarts_optimizer=10,
            random_state=42
        )
        self.gp_time = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(),
            n_restarts_optimizer=10,
            random_state=42
        )
        self.gp_memory = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(),
            n_restarts_optimizer=10,
            random_state=42
        )
        
        self.scaler_X = StandardScaler()
        self.scaler_y = {
            'psnr': StandardScaler(),
            'time': StandardScaler(),
            'memory': StandardScaler()
        }
        
        self.X_train = None
        self.y_train = {
            'psnr': [],
            'time': [],
            'memory': []
        }

    def fit(self, X, y_dict):
        """Fit the surrogate models"""
        self.X_train = self.scaler_X.fit_transform(X)
        
        for metric, values in y_dict.items():
            scaled_y = self.scaler_y[metric].fit_transform(np.array(values).reshape(-1, 1))
            self.y_train[metric] = scaled_y.ravel()
            
        self.gp_psnr.fit(self.X_train, self.y_train['psnr'])
        self.gp_time.fit(self.X_train, self.y_train['time'])
        self.gp_memory.fit(self.X_train, self.y_train['memory'])

    def predict(self, X):
        """Predict objectives using the surrogate models"""
        X_scaled = self.scaler_X.transform(X)
        
        psnr_pred = self.gp_psnr.predict(X_scaled)
        time_pred = self.gp_time.predict(X_scaled)
        memory_pred = self.gp_memory.predict(X_scaled)
        
        # Unscale predictions
        psnr_pred = self.scaler_y['psnr'].inverse_transform(psnr_pred.reshape(-1, 1)).ravel()
        time_pred = self.scaler_y['time'].inverse_transform(time_pred.reshape(-1, 1)).ravel()
        memory_pred = self.scaler_y['memory'].inverse_transform(memory_pred.reshape(-1, 1)).ravel()
        
        return psnr_pred, time_pred, memory_pred

def evaluate_individual_real(individual):
    """Evaluate individual using actual NeRF training"""
    hyperparams = {
        'learning_rate': individual[0],
        'num_hidden_layers': int(round(individual[1])),
        'neurons_per_layer': int(round(individual[2])),
        'batch_size': int(round(individual[3])),
        'learning_decay_steps': int(round(individual[4]))
    }
    
    psnr, train_time, memory, _ = evaluate_hyperparameters(hyperparams)
    cpu_usage = psutil.cpu_percent()
    
    return psnr, train_time, cpu_usage, memory

def evaluate_individual_surrogate(individual, surrogate):
    """Evaluate individual using surrogate model"""
    X = np.array(individual).reshape(1, -1)
    psnr_pred, time_pred, memory_pred = surrogate.predict(X)
    cpu_pred = time_pred * 0.5  # Rough estimation of CPU usage
    
    return psnr_pred[0], time_pred[0], cpu_pred, memory_pred[0]

def initialize_population():
    """Initialize population with bounded random values"""
    ind = creator.Individual([
        random.uniform(1e-5, 1e-2),  # learning_rate
        random.uniform(1, 10),       # num_hidden_layers
        random.uniform(32, 512),     # neurons_per_layer
        random.uniform(16, 1024),    # batch_size
        random.uniform(100, 1000)    # learning_decay_steps
    ])
    return ind

def main():
    # Initialize DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", initialize_population)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Parameters
    NGEN = 50
    POPSIZE = 20
    INITIAL_REAL_EVALS = 10
    REAL_EVALS_PER_GEN = 2
    
    # Initialize surrogate model
    surrogate = SurrogateModel()
    
    # Initial population with real evaluations
    population = toolbox.population(n=INITIAL_REAL_EVALS)
    X_initial = []
    y_psnr = []
    y_time = []
    y_memory = []
    
    print("Performing initial real evaluations...")
    for ind in population:
        psnr, time, cpu, memory = evaluate_individual_real(ind)
        X_initial.append(ind)
        y_psnr.append(psnr)
        y_time.append(time)
        y_memory.append(memory)
        ind.fitness.values = (psnr, time, cpu, memory)
    
    # Fit surrogate model
    surrogate.fit(
        np.array(X_initial),
        {'psnr': y_psnr, 'time': y_time, 'memory': y_memory}
    )
    
    # Evolution loop
    for gen in range(NGEN):
        print(f"\nGeneration {gen}")
        
        # Generate offspring using surrogate model
        offspring = algorithms.varOr(
            population, toolbox, lambda_=POPSIZE, 
            cxpb=0.7, mutpb=0.2
        )
        
        # Evaluate offspring using surrogate
        for ind in offspring:
            if not ind.fitness.valid:
                surrogate_fitness = evaluate_individual_surrogate(ind, surrogate)
                ind.fitness.values = surrogate_fitness
        
        # Select best individuals for real evaluation
        combined_pop = population + offspring
        fronts = tools.sortNondominated(combined_pop, len(combined_pop))
        to_evaluate = []
        
        for front in fronts:
            if len(to_evaluate) + len(front) <= REAL_EVALS_PER_GEN:
                to_evaluate.extend(front)
            else:
                break
        
        # Perform real evaluations and update surrogate
        X_new = []
        y_psnr_new = []
        y_time_new = []
        y_memory_new = []
        
        for ind in to_evaluate:
            psnr, time, cpu, memory = evaluate_individual_real(ind)
            X_new.append(ind)
            y_psnr_new.append(psnr)
            y_time_new.append(time)
            y_memory_new.append(memory)
            ind.fitness.values = (psnr, time, cpu, memory)
        
        # Update surrogate model
        if X_new:
            X_initial.extend(X_new)
            y_psnr.extend(y_psnr_new)
            y_time.extend(y_time_new)
            y_memory.extend(y_memory_new)
            
            surrogate.fit(
                np.array(X_initial),
                {'psnr': y_psnr, 'time': y_time, 'memory': y_memory}
            )
        
        # Select next generation
        population = tools.selNSGA2(combined_pop, POPSIZE)
        
        # Print current best solutions
        fronts = tools.sortNondominated(population, len(population))
        print("\nPareto front:")
        for ind in fronts[0]:
            print(f"PSNR: {ind.fitness.values[0]:.2f}, "
                  f"Time: {ind.fitness.values[1]:.2f}, "
                  f"CPU: {ind.fitness.values[2]:.2f}, "
                  f"Memory: {ind.fitness.values[3]:.2f}")
            print(f"Hyperparameters: lr={ind[0]:.6f}, "
                  f"layers={int(round(ind[1]))}, "
                  f"neurons={int(round(ind[2]))}, "
                  f"batch={int(round(ind[3]))}, "
                  f"decay={int(round(ind[4]))}")

if __name__ == "__main__":
    main()