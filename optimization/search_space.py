import numpy as np

# Define hyperparameter bounds
HYPERPARAMETER_BOUNDS = {
    "netdepth": (4, 12),         
    "netwidth": (64, 256),       
    "lrate": (1e-4, 1e-2),      
    "N_rand": (512, 1024)       
}

BOUNDS = np.array([HYPERPARAMETER_BOUNDS[key] for key in HYPERPARAMETER_BOUNDS.keys()])
