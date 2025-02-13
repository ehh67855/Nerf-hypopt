import subprocess
import json
import time
import os

# Path to `run_nerf.py` inside `nerf-pytorch`
NERF_SCRIPT = os.path.join(os.path.dirname(__file__), "../nerf-pytorch/run_nerf.py")

def train_nerf(netdepth, netwidth, lrate, N_rand, config_path):
    """
    Trains a NeRF model with the given hyperparameters and returns performance metrics.

    Args:
        netdepth (int): Number of hidden layers in MLP.
        netwidth (int): Number of neurons per layer.
        lrate (float): Learning rate for training.
        N_rand (int): Number of random rays per batch.
        config_path (str): Path to the configuration file.

    Returns:
        tuple: (PSNR, training time, memory usage)
    """
    start_time = time.time()

    # Construct the command to run NeRF
    train_cmd = [
        "python", NERF_SCRIPT,
        "--netdepth", str(netdepth),
        "--netwidth", str(netwidth),
        "--lrate", str(lrate),
        "--N_rand", str(N_rand),
        "--config", config_path
    ]

    try:
        # Execute the training command
        process = subprocess.run(train_cmd, capture_output=True, text=True, check=True)

        # Extract output
        output_lines = process.stdout.split("\n")
        metrics = None

        for line in output_lines:
            if "PSNR:" in line:  # Modify if needed based on NeRF's log output
                try:
                    metrics = json.loads(line)  # Expected output: {"PSNR": 30.5, "train_time": 500, "gpu_memory": 2.5}
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON from line: {line}")

        if metrics:
            psnr = metrics.get("PSNR", 0)
            memory_usage = metrics.get("gpu_memory", float("inf"))
        else:
            psnr, memory_usage = 0, float("inf")

    except subprocess.CalledProcessError as e:
        print(f"Error running NeRF training: {e}")
        psnr, memory_usage = 0, float("inf")

    train_time = time.time() - start_time  # Measure total execution time

    return psnr, train_time, memory_usage


if __name__ == "__main__":
    # Example usage
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "../nerf-pytorch/configs/lego.txt")

    psnr, train_time, memory = train_nerf(10, 512, 0.001, 512, CONFIG_FILE)

    print(f"Training Results: PSNR={psnr}, Time={train_time:.2f}s, Memory={memory}GB")

