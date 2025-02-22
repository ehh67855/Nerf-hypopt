import torch
import os
import subprocess
import json

# Path to `run_nerf.py`
NERF_SCRIPT = os.path.join(os.path.dirname(__file__), "../nerf-pytorch/run_nerf.py")

def evaluate_nerf(model_path, config_path):
    """
    Evaluates a trained NeRF model.

    Args:
        model_checkpoint (str): Path to the trained model checkpoint (e.g., model.pth).
        config_path (str): Path to the configuration file.
    Returns:
        float: PSNR value of the evaluation.
    """
    # Ensure the checkpoint exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Construct evaluation command
    eval_cmd = [
        "python", NERF_SCRIPT,
        "--config", config_path,
        "--render_only", "True",  # Flag to render evaluation images
        "--ft_path", model_checkpoint  # Load trained model
    ]

    try:
        process = subprocess.run(eval_cmd, capture_output=True, text=True, check=True, stderr=subprocess.STDOUT)
        output_lines = process.stdout.split("\n")

        psnr = 0
        for line in output_lines:
            if "PSNR:" in line:  # Modify based on NeRF's evaluation output
                try:
                    metrics = json.loads(line)  # Expected format: {"PSNR": value}
                    psnr = metrics.get("PSNR", 0)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON from line: {line}")

    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        print(e.output)
        psnr, memory_usage = 0, float("inf")
    
    return psnr, memeory_usage


if __name__ == "__main__":
    # Define paths
    MODEL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "../nerf-pytorch/logs/lego/model.pth")
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "../nerf-pytorch/configs/lego.txt")

    # Run evaluation
    psnr = evaluate_nerf(MODEL_CHECKPOINT, CONFIG_FILE)

    print(f"Evaluation PSNR: {psnr}")
