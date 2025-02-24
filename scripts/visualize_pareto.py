import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots

def visualize_pareto(results):
    """
    Visualizes the Pareto front for PSNR vs. Memory Usage.

    Args:
        results (list of tuples): List of (PSNR, memory_usage) tuples.
    """
    # Extract PSNR and memory usage from results
    psnrs = [result[0] for result in results]
    memory_usages = [result[1] for result in results]

    # Create a 2D scatter plot using matplotlib
    plt.scatter(memory_usages, psnrs)
    plt.xlabel('Memory Usage (GB)')
    plt.ylabel('PSNR')
    plt.title('Pareto Front: PSNR vs Memory Usage')
    plt.show()

def visualize_multi_objective(results):
    """
    Visualizes multi-objective optimization results in 3D (PSNR, Memory Usage, Training Time).

    Args:
        results (list of tuples): List of (PSNR, memory_usage, train_time) tuples.
    """
    # Extract PSNR, memory usage, and training time from results
    psnrs = [result[0] for result in results]
    memory_usages = [result[1] for result in results]
    train_times = [result[2] for result in results]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(memory_usages, train_times, psnrs)
    ax.set_xlabel('Memory Usage (GB)')
    ax.set_ylabel('Training Time (s)')
    ax.set_zlabel('PSNR')
    plt.title('Optimization Results')
    plt.show()


if __name__ == "__main__":
    
    # 2D visualization
    visualize_pareto([(r[0], r[1]) for r in results])   
    # 3D visualization
    visualize_multi_objective(results)  
