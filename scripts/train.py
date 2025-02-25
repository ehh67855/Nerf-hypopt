import os
import subprocess
import json
import time
import psutil
import torch

# Get absolute paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NERF_SCRIPT = os.path.abspath(os.path.join(REPO_ROOT, "nerf-pytorch", "run_nerf.py"))

def train_nerf(hyperparams):
    """
    Trains NeRF with given hyperparameters and returns metrics.
    
    Args:
        hyperparams (list): [netdepth, netwidth, lrate, N_rand]
    
    Returns:
        tuple: (PSNR score, training time in seconds, peak memory usage in MB)
    """
    netdepth, netwidth, lrate, N_rand = hyperparams
    
    # Convert hyperparameters to appropriate types
    netdepth = int(netdepth)
    netwidth = int(netwidth)
    N_rand = int(N_rand)
    
    # Verify script exists
    if not os.path.exists(NERF_SCRIPT):
        raise FileNotFoundError(f"NeRF script not found at: {NERF_SCRIPT}")
    
    cmd = [
        "python",
        NERF_SCRIPT,
        "--datadir", "./data/nerf_synthetic/lego",  
        "--dataset_type", "blender",                
        "--netdepth", str(netdepth),
        "--netwidth", str(netwidth),
        "--lrate", str(lrate),
        "--N_rand", str(N_rand),
        "--no_reload"                             
    ]
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    try:
        # Run NeRF training
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract PSNR from output (modify based on actual output format)
        for line in result.stdout.split('\n'):
            if 'PSNR' in line:
                psnr = float(line.split(':')[1].strip())
                break
        else:
            psnr = 0.0  # Default if PSNR not found
        
        # Calculate metrics
        train_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = final_memory - initial_memory
        
        return psnr, train_time, memory_usage
        
    except subprocess.CalledProcessError as e:
        print(f"Error running NeRF training: {e}")
        print(f"stderr: {e.stderr}")
        return 0.0, 999999, 999999  # Return poor fitness values on error

if __name__ == "__main__":
    # Example usage
    hyperparams = {
        "netdepth": 10,
        "netwidth": 512,
        "lrate": 0.001,
        "N_rand": 512
    }
    
    config_path = os.path.join(os.path.dirname(__file__), 
                              "../nerf-pytorch/configs/lego.txt")
    
    results = train_nerf(hyperparams)
    print(f"Training Results:")
    print(f"PSNR: {results[0]:.2f}")
    print(f"Training Time: {results[1]:.2f} seconds")
    print(f"Peak Memory Usage: {results[2]:.2f} MB")

