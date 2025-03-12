import os
import subprocess
import json
import time
import psutil
import torch

# Get absolute paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NERF_SCRIPT = os.path.abspath(os.path.join(REPO_ROOT, "nerf-pytorch", "run_nerf.py"))

def train_nerf(hyperparams, config_path):
    """
    Trains NeRF with given hyperparameters and returns metrics.
    
    Args:
        hyperparams (dict): Dictionary of hyperparameters
        config_path (str): Path to base config file
    
    Returns:
        dict: Dictionary containing PSNR, training time, and peak memory usage
    """
    # Convert config_path to absolute path if it isn't already
    config_path = os.path.abspath(config_path)
    
    # Add these debug prints at the start of the function
    print(f"NERF_SCRIPT path: {NERF_SCRIPT}")
    print(f"Config path: {config_path}")
    print(f"Checking if paths exist:")
    print(f"NERF_SCRIPT exists: {os.path.exists(NERF_SCRIPT)}")
    print(f"Config exists: {os.path.exists(config_path)}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Construct training command
    train_cmd = [
        "python", NERF_SCRIPT,
        "--config", config_path,
        "--netdepth", str(hyperparams.get('netdepth', 8)),
        "--netwidth", str(hyperparams.get('netwidth', 256)),
        "--lrate", str(hyperparams.get('lrate', 5e-4)),
        "--N_rand", str(hyperparams.get('N_rand', 1024))
    ]

    start_time = time.time()
    current_process = psutil.Process(os.getpid())  # Rename this variable
    peak_memory = 0

    try:
        # Change to the NeRF directory before running
        original_dir = os.getcwd()
        nerf_dir = os.path.dirname(NERF_SCRIPT)
        os.chdir(nerf_dir)
        
        # Store initial memory usage
        initial_memory = current_process.memory_info().rss / 1024 / 1024  # MB
        
        print("\nStarting NeRF training with following command:")
        print(" ".join(train_cmd))
        print("\nTraining output:")
        
        # Use Popen to get real-time output while also capturing it
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Initialize variables to store output
        output_lines = []
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())
        
        # Get the return code
        return_code = process.poll()
        
        # Track final memory usage
        final_memory = current_process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = max(initial_memory, final_memory)
        
        # Change back to original directory
        os.chdir(original_dir)
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, train_cmd)
        
        # Parse output for metrics
        final_psnr = 0
        for line in output_lines:
            if "PSNR" in line:
                try:
                    metrics = json.loads(line)
                    final_psnr = metrics.get("PSNR", 0)
                except json.JSONDecodeError:
                    continue

    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        print(e.output)
        return {
            "psnr": 0,
            "training_time": float("inf"),
            "peak_memory_mb": float("inf")
        }

    training_time = time.time() - start_time

    return {
        "psnr": final_psnr,
        "training_time": training_time,
        "peak_memory_mb": peak_memory
    }

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
    
    results = train_nerf(hyperparams, config_path)
    print(f"Training Results:")
    print(f"PSNR: {results['psnr']:.2f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Peak Memory Usage: {results['peak_memory_mb']:.2f} MB")
