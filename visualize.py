import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Create directories to save plots
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load optimization logs from json files
log_dir = "logs/optimization"  # directory containing JSON files
all_data = []
for file in sorted(os.listdir(log_dir)):
    if file.startswith("gen_") and file.endswith(".json"):
        with open(os.path.join(log_dir, file)) as f:
            gen_data = json.load(f)
            for entry in gen_data:
                # Flatten the params dictionary and combine it with other fields
                flattened_entry = {
                    **entry['params'],  # Flatten params into top-level keys
                    'gen': entry['gen'],
                    'index': entry['index'],
                    'fitness_psnr': entry['fitness'][0],  # PSNR
                    'fitness_time': entry['fitness'][1],  # Training Time
                    'fitness_memory': entry['fitness'][2],  # Memory Usage
                    'real_eval': entry['real_eval']
                }
                all_data.append(flattened_entry)

# Convert data to dataframe for easier manipulation
df = pd.DataFrame(all_data)

# Extract hyperparameter columns (excluding the metrics and metadata)
param_cols = [col for col in df.columns if col not in 
              ['gen', 'index', 'fitness_psnr', 'fitness_time', 'fitness_memory', 'real_eval']]

# Print information about the dataset
print("DataFrame info:")
print(f"Total individuals: {len(df)}")
print(f"Number of generations: {df['gen'].nunique()}")
print(f"Hyperparameters: {param_cols}")
print(f"Fitness range - PSNR: {df['fitness_psnr'].min():.2f} to {df['fitness_psnr'].max():.2f}")
print(f"Fitness range - Time: {df['fitness_time'].min():.2f} to {df['fitness_time'].max():.2f}")
print(f"Fitness range - Memory: {df['fitness_memory'].min():.2f} to {df['fitness_memory'].max():.2f}")

# 1. PARETO FRONT VISUALIZATIONS
def plot_pareto_front(df):
    """Plot 2D and 3D Pareto fronts"""
    # 2D Pareto Front: PSNR vs. Time
    plt.figure(figsize=(10, 8))
    real_evals = df[df['real_eval'] == True]
    surrogate_evals = df[df['real_eval'] == False]
    
    # Plot with distinction between real and surrogate evaluations
    plt.scatter(surrogate_evals['fitness_psnr'], surrogate_evals['fitness_time'], 
                c=surrogate_evals['gen'], cmap='viridis', s=50, alpha=0.5, label='Surrogate')
    plt.scatter(real_evals['fitness_psnr'], real_evals['fitness_time'], 
                c=real_evals['gen'], cmap='viridis', s=80, edgecolors='black', label='Real')
    
    # Add colorbar for generation
    cbar = plt.colorbar()
    cbar.set_label('Generation')
    
    # Identify and highlight true Pareto front (from real evaluations only)
    is_pareto = np.ones(len(real_evals), dtype=bool)
    for i, (psnr_i, time_i) in enumerate(zip(real_evals['fitness_psnr'], real_evals['fitness_time'])):
        # Check if any point dominates this point
        for psnr_j, time_j in zip(real_evals['fitness_psnr'], real_evals['fitness_time']):
            if (psnr_j > psnr_i and time_j <= time_i) or (psnr_j >= psnr_i and time_j < time_i):
                is_pareto[i] = False
                break
    
    # Draw lines connecting Pareto optimal points
    pareto_points = real_evals[is_pareto].sort_values('fitness_psnr')
    plt.plot(pareto_points['fitness_psnr'], pareto_points['fitness_time'], 
             'r--', linewidth=2, label='Pareto Front')
    
    plt.xlabel('PSNR (higher is better)')
    plt.ylabel('Training Time (lower is better)')
    plt.title('Pareto Front: PSNR vs. Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pareto_front_2d_psnr_time.png"))
    
    # 2D Pareto Front: PSNR vs. Memory
    plt.figure(figsize=(10, 8))
    plt.scatter(surrogate_evals['fitness_psnr'], surrogate_evals['fitness_memory'], 
                c=surrogate_evals['gen'], cmap='viridis', s=50, alpha=0.5, label='Surrogate')
    plt.scatter(real_evals['fitness_psnr'], real_evals['fitness_memory'], 
                c=real_evals['gen'], cmap='viridis', s=80, edgecolors='black', label='Real')
    
    # Add colorbar for generation
    cbar = plt.colorbar()
    cbar.set_label('Generation')
    
    # Identify and highlight Pareto front for PSNR vs. Memory
    is_pareto_mem = np.ones(len(real_evals), dtype=bool)
    for i, (psnr_i, mem_i) in enumerate(zip(real_evals['fitness_psnr'], real_evals['fitness_memory'])):
        for psnr_j, mem_j in zip(real_evals['fitness_psnr'], real_evals['fitness_memory']):
            if (psnr_j > psnr_i and mem_j <= mem_i) or (psnr_j >= psnr_i and mem_j < mem_i):
                is_pareto_mem[i] = False
                break
    
    # Draw lines connecting Pareto optimal points
    pareto_points_mem = real_evals[is_pareto_mem].sort_values('fitness_psnr')
    plt.plot(pareto_points_mem['fitness_psnr'], pareto_points_mem['fitness_memory'], 
             'r--', linewidth=2, label='Pareto Front')
    
    plt.xlabel('PSNR (higher is better)')
    plt.ylabel('Memory Usage (lower is better)')
    plt.title('Pareto Front: PSNR vs. Memory')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pareto_front_2d_psnr_memory.png"))
    
    # 3D Pareto Front: PSNR vs. Time vs. Memory
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot with distinction between real and surrogate evaluations
    scatter1 = ax.scatter(surrogate_evals['fitness_psnr'], 
                          surrogate_evals['fitness_time'], 
                          surrogate_evals['fitness_memory'],
                          c=surrogate_evals['gen'], cmap='viridis', s=30, alpha=0.5, label='Surrogate')
    scatter2 = ax.scatter(real_evals['fitness_psnr'], 
                          real_evals['fitness_time'], 
                          real_evals['fitness_memory'],
                          c=real_evals['gen'], cmap='viridis', s=50, edgecolors='black', label='Real')
    
    # Add colorbar for generation
    cbar = fig.colorbar(scatter1, ax=ax, shrink=0.6)
    cbar.set_label('Generation')
    
    # Identify 3D Pareto front
    is_pareto_3d = np.ones(len(real_evals), dtype=bool)
    for i, (psnr_i, time_i, mem_i) in enumerate(zip(
        real_evals['fitness_psnr'], real_evals['fitness_time'], real_evals['fitness_memory'])):
        for psnr_j, time_j, mem_j in zip(
            real_evals['fitness_psnr'], real_evals['fitness_time'], real_evals['fitness_memory']):
            # Check if point j dominates point i
            if ((psnr_j > psnr_i and time_j <= time_i and mem_j <= mem_i) or
                (psnr_j >= psnr_i and time_j < time_i and mem_j <= mem_i) or
                (psnr_j >= psnr_i and time_j <= time_i and mem_j < mem_i)):
                is_pareto_3d[i] = False
                break
    
    # Highlight Pareto optimal points
    pareto_points_3d = real_evals[is_pareto_3d]
    ax.scatter(pareto_points_3d['fitness_psnr'], 
               pareto_points_3d['fitness_time'], 
               pareto_points_3d['fitness_memory'],
               color='red', s=100, edgecolors='black', marker='*', label='Pareto Optimal')
    
    ax.set_xlabel('PSNR (higher is better)')
    ax.set_ylabel('Training Time (lower is better)')
    ax.set_zlabel('Memory Usage (lower is better)')
    ax.set_title('3D Pareto Front: PSNR vs. Time vs. Memory')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pareto_front_3d.png"))
    plt.close('all')

def animate_pareto_front(df):
    """Create animation of Pareto front evolution over generations"""
    generations = sorted(df['gen'].unique())
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_surr = ax.scatter([], [], c=[], cmap='viridis', alpha=0.5, s=50, label='Surrogate')
    scatter_real = ax.scatter([], [], c='red', edgecolors='black', s=80, label='Real')
    line, = ax.plot([], [], 'r--', linewidth=2, label='Pareto Front')
    
    ax.set_xlim(df['fitness_psnr'].min() - 1, df['fitness_psnr'].max() + 1)
    ax.set_ylim(df['fitness_time'].min() - 0.1, df['fitness_time'].max() + 0.1)
    ax.set_xlabel('PSNR (higher is better)')
    ax.set_ylabel('Training Time (lower is better)')
    ax.legend(loc='upper right')
    
    title = ax.set_title('')
    
    # Define animation update function
    def update(frame):
        # Get data up to the current generation
        current_data = df[df['gen'] <= generations[frame]]
        
        # Split data into real and surrogate evaluations
        real_data = current_data[current_data['real_eval'] == True]
        surr_data = current_data[current_data['real_eval'] == False]
        
        # Update scatter plots
        scatter_surr.set_offsets(np.c_[surr_data['fitness_psnr'], surr_data['fitness_time']])
        scatter_surr.set_array(surr_data['gen'])
        scatter_real.set_offsets(np.c_[real_data['fitness_psnr'], real_data['fitness_time']])
        
        # Calculate Pareto front
        if len(real_data) > 0:
            is_pareto = np.ones(len(real_data), dtype=bool)
            for i, (psnr_i, time_i) in enumerate(zip(real_data['fitness_psnr'], real_data['fitness_time'])):
                for psnr_j, time_j in zip(real_data['fitness_psnr'], real_data['fitness_time']):
                    if (psnr_j > psnr_i and time_j <= time_i) or (psnr_j >= psnr_i and time_j < time_i):
                        is_pareto[i] = False
                        break
            
            # Update Pareto front line
            pareto_points = real_data[is_pareto].sort_values('fitness_psnr')
            if len(pareto_points) > 1:
                line.set_data(pareto_points['fitness_psnr'], pareto_points['fitness_time'])
            else:
                line.set_data([], [])
        else:
            line.set_data([], [])
        
        title.set_text(f'Pareto Front Evolution - Generation {generations[frame]}')
        return scatter_surr, scatter_real, line, title
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(generations), interval=500, blit=False)
    anim.save(os.path.join(plots_dir, "pareto_front_animation.mp4"), fps=4, dpi=150)
    plt.close(fig)
    print(f"Animation saved as pareto_front_animation.mp4")

# 2. FITNESS EVOLUTION PLOTS
def plot_fitness_evolution(df):
    """Plot evolution of fitness metrics over generations"""
    generations = sorted(df['gen'].unique())
    
    # Set up figure with 3 subplots (one for each metric)
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # Calculate statistics for each generation
    metrics = ['fitness_psnr', 'fitness_time', 'fitness_memory']
    titles = ['PSNR Evolution (higher is better)', 
              'Training Time Evolution (lower is better)', 
              'Memory Usage Evolution (lower is better)']
    colors = ['green', 'blue', 'purple']
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        best_vals = []
        avg_vals = []
        worst_vals = []
        
        # For PSNR, higher is better; for time and memory, lower is better
        is_higher_better = (metric == 'fitness_psnr')
        
        for gen in generations:
            gen_data = df[df['gen'] == gen]
            if is_higher_better:
                best_vals.append(gen_data[metric].max())
                worst_vals.append(gen_data[metric].min())
            else:
                best_vals.append(gen_data[metric].min())
                worst_vals.append(gen_data[metric].max())
            avg_vals.append(gen_data[metric].mean())
        
        # Plot the data
        axs[i].plot(generations, best_vals, label='Best', color=color, linestyle='-', linewidth=2, marker='o')
        axs[i].plot(generations, avg_vals, label='Average', color=color, linestyle='--', linewidth=1.5)
        axs[i].plot(generations, worst_vals, label='Worst', color=color, linestyle=':', linewidth=1)
        axs[i].fill_between(generations, worst_vals, best_vals, color=color, alpha=0.1)
        
        axs[i].set_title(title)
        axs[i].set_ylabel(metric.split('_')[1].upper())
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].legend()
    
    axs[2].set_xlabel('Generation')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fitness_evolution.png"))
    plt.close(fig)

# 3. DIVERSITY ANALYSIS
def plot_population_diversity(df):
    """Visualize diversity of individuals across generations"""
    # 3.1. Box plots of fitness values per generation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['fitness_psnr', 'fitness_time', 'fitness_memory']
    titles = ['PSNR Distribution', 'Training Time Distribution', 'Memory Usage Distribution']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.boxplot(x='gen', y=metric, data=df, ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xlabel('Generation')
        axes[i].set_ylabel(metric.split('_')[1].upper())
        
        # Rotate x-axis labels if many generations
        if df['gen'].nunique() > 10:
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fitness_distributions.png"))
    plt.close(fig)
    
    # 3.2. PCA visualization of parameter space
    # Extract hyperparameters for dimensionality reduction
    param_data = df[param_cols].copy()
    
    # Normalize parameters to [0, 1] range for better PCA projection
    for col in param_data.columns:
        min_val = param_data[col].min()
        max_val = param_data[col].max()
        if max_val > min_val:  # Avoid division by zero
            param_data[col] = (param_data[col] - min_val) / (max_val - min_val)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(param_data.values)
    
    # Create PCA scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=df['gen'], cmap='viridis', s=70, alpha=0.7)
    plt.colorbar(scatter, label='Generation')
    
    # Add arrows to show component directions
    if len(param_cols) <= 8:  # Only show arrows if not too many params
        for i, feature in enumerate(param_cols):
            plt.arrow(0, 0, pca.components_[0, i]*5, pca.components_[1, i]*5,
                      head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
            plt.text(pca.components_[0, i]*5.2, pca.components_[1, i]*5.2, feature)
    
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Projection of Hyperparameter Space')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hyperparameter_pca.png"))
    plt.close()
    
    # 3.3. Pairwise distance heatmap
    generations = sorted(df['gen'].unique())
    if len(generations) > 1:  # Only proceed if there are multiple generations
        avg_distances = np.zeros((len(generations), len(generations)))
        
        for i, gen_i in enumerate(generations):
            for j, gen_j in enumerate(generations):
                # Get normalized parameters for each generation
                params_i = df[df['gen'] == gen_i][param_cols].values
                params_j = df[df['gen'] == gen_j][param_cols].values
                
                # Compute all pairwise distances between individuals in both generations
                distances = []
                for p1 in params_i:
                    for p2 in params_j:
                        distances.append(np.sqrt(np.sum((p1 - p2)**2)))
                
                avg_distances[i, j] = np.mean(distances)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_distances, annot=True, cmap='YlGnBu', 
                    xticklabels=generations, yticklabels=generations)
        plt.title('Average Pairwise Distances Between Generations')
        plt.xlabel('Generation')
        plt.ylabel('Generation')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "generation_distances.png"))
        plt.close()

# 4. SURROGATE MODEL ACCURACY
def plot_surrogate_accuracy(df):
    """Visualize surrogate model prediction accuracy"""
    # Check if we have both real evaluations and surrogate predictions
    real_evals = df[df['real_eval'] == True]
    surrogate_evals = df[df['real_eval'] == False]
    
    if len(real_evals) == 0 or len(surrogate_evals) == 0:
        print("Insufficient data for surrogate model accuracy plots")
        return
    
    # Find matching individuals (same hyperparameters but different evaluation types)
    # This is an approximation as we may not have exact matches
    param_cols = [col for col in df.columns if col not in 
                  ['gen', 'index', 'fitness_psnr', 'fitness_time', 'fitness_memory', 'real_eval']]
    
    # Group by generation and find matches
    generations = sorted(df['gen'].unique())
    
    # Collect predicted vs actual values across all metrics
    predicted_psnr = []
    actual_psnr = []
    predicted_time = []
    actual_time = []
    predicted_memory = []
    actual_memory = []
    
    # R² and RMSE over generations
    r2_over_time = {'psnr': [], 'time': [], 'memory': []}
    rmse_over_time = {'psnr': [], 'time': [], 'memory': []}
    
    # For each generation
    for gen in generations:
        gen_real = real_evals[real_evals['gen'] <= gen]  # All real evals up to this generation
        gen_surr = surrogate_evals[surrogate_evals['gen'] == gen]  # Current gen surrogate evals
        
        if len(gen_real) == 0 or len(gen_surr) == 0:
            continue
        
        # Find closest real evaluation for each surrogate evaluation
        for idx, surr_row in gen_surr.iterrows():
            # Extract parameter values for this surrogate evaluation
            surr_params = surr_row[param_cols].values
            
            # Calculate distance to all real evaluations
            distances = []
            for real_idx, real_row in gen_real.iterrows():
                real_params = real_row[param_cols].values
                dist = np.sqrt(np.sum((surr_params - real_params)**2))
                distances.append((dist, real_idx))
            
            # Find closest real evaluation
            distances.sort()
            if len(distances) > 0:
                closest_real_idx = distances[0][1]
                closest_real = df.loc[closest_real_idx]
                
                # Only use very close matches (adjust threshold as needed)
                if distances[0][0] < 0.1:  # Small distance threshold
                    predicted_psnr.append(surr_row['fitness_psnr'])
                    actual_psnr.append(closest_real['fitness_psnr'])
                    predicted_time.append(surr_row['fitness_time'])
                    actual_time.append(closest_real['fitness_time'])
                    predicted_memory.append(surr_row['fitness_memory'])
                    actual_memory.append(closest_real['fitness_memory'])
        
        # Calculate metrics for this generation
        if len(predicted_psnr) > 0:
            r2_over_time['psnr'].append(r2_score(actual_psnr, predicted_psnr))
            rmse_over_time['psnr'].append(np.sqrt(mean_squared_error(actual_psnr, predicted_psnr)))
        else:
            r2_over_time['psnr'].append(0)
            rmse_over_time['psnr'].append(0)
            
        if len(predicted_time) > 0:
            r2_over_time['time'].append(r2_score(actual_time, predicted_time))
            rmse_over_time['time'].append(np.sqrt(mean_squared_error(actual_time, predicted_time)))
        else:
            r2_over_time['time'].append(0)
            rmse_over_time['time'].append(0)
            
        if len(predicted_memory) > 0:
            r2_over_time['memory'].append(r2_score(actual_memory, predicted_memory))
            rmse_over_time['memory'].append(np.sqrt(mean_squared_error(actual_memory, predicted_memory)))
        else:
            r2_over_time['memory'].append(0)
            rmse_over_time['memory'].append(0)
    
    # Plot predicted vs. true values
    if len(predicted_psnr) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # PSNR
        axes[0].scatter(actual_psnr, predicted_psnr, alpha=0.7)
        axes[0].plot([min(actual_psnr), max(actual_psnr)], [min(actual_psnr), max(actual_psnr)], 'r--')
        axes[0].set_xlabel('True PSNR')
        axes[0].set_ylabel('Predicted PSNR')
        axes[0].set_title(f'PSNR: R² = {r2_score(actual_psnr, predicted_psnr):.3f}')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Time
        axes[1].scatter(actual_time, predicted_time, alpha=0.7)
        axes[1].plot([min(actual_time), max(actual_time)], [min(actual_time), max(actual_time)], 'r--')
        axes[1].set_xlabel('True Training Time')
        axes[1].set_ylabel('Predicted Training Time')
        axes[1].set_title(f'Time: R² = {r2_score(actual_time, predicted_time):.3f}')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Memory
        axes[2].scatter(actual_memory, predicted_memory, alpha=0.7)
        axes[2].plot([min(actual_memory), max(actual_memory)], [min(actual_memory), max(actual_memory)], 'r--')
        axes[2].set_xlabel('True Memory Usage')
        axes[2].set_ylabel('Predicted Memory Usage')
        axes[2].set_title(f'Memory: R² = {r2_score(actual_memory, predicted_memory):.3f}')
        axes[2].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "surrogate_accuracy.png"))
        plt.close(fig)
        
        # Plot R² and RMSE over time
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # R² evolution
        axes[0].plot(generations[:len(r2_over_time['psnr'])], r2_over_time['psnr'], 
                    label='PSNR', marker='o', color='blue')
        axes[0].plot(generations[:len(r2_over_time['time'])], r2_over_time['time'], 
                    label='Time', marker='s', color='green')
        axes[0].plot(generations[:len(r2_over_time['memory'])], r2_over_time['memory'], 
                    label='Memory', marker='^', color='red')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Surrogate Model R² Score Over Generations')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend()
        
        # RMSE evolution (normalized to [0,1] range for comparison)
        # Normalize RMSE values for better comparison
        max_rmse_psnr = max(rmse_over_time['psnr']) if max(rmse_over_time['psnr']) > 0 else 1
        max_rmse_time = max(rmse_over_time['time']) if max(rmse_over_time['time']) > 0 else 1
        max_rmse_memory = max(rmse_over_time['memory']) if max(rmse_over_time['memory']) > 0 else 1
        
        axes[1].plot(generations[:len(rmse_over_time['psnr'])], 
                    [x/max_rmse_psnr for x in rmse_over_time['psnr']], 
                    label='PSNR', marker='o', color='blue')
        axes[1].plot(generations[:len(rmse_over_time['time'])], 
                    [x/max_rmse_time for x in rmse_over_time['time']], 
                    label='Time', marker='s', color='green')
        axes[1].plot(generations[:len(rmse_over_time['memory'])], 
                    [x/max_rmse_memory for x in rmse_over_time['memory']], 
                    label='Memory', marker='^', color='red')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Normalized RMSE')
        axes[1].set_title('Surrogate Model Normalized RMSE Over Generations')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "surrogate_performance_over_time.png"))
        plt.close(fig)
 # 5. GAUSSIAN PROCESS UNCERTAINTY VISUALIZATION
def visualize_gp_uncertainty(df):
    """
    Visualize surrogate model uncertainty (simulating Gaussian Process behavior)
    Since we don't have direct access to the GP model, we'll simulate uncertainty
    based on data density in parameter space
    """
    # Find top 2 most important parameters (using variance as a proxy)
    param_variances = [(col, df[col].var()) for col in param_cols]
    param_variances.sort(key=lambda x: x[1], reverse=True)
    
    if len(param_variances) < 2:
        print("Not enough parameters for GP uncertainty visualization")
        return
    
    top_params = [param_variances[0][0], param_variances[1][0]]
    print(f"Using top parameters for GP uncertainty: {top_params}")
    
    # Create grid for visualization (2D grid of top two parameters)
    param1_range = np.linspace(df[top_params[0]].min(), df[top_params[0]].max(), 50)
    param2_range = np.linspace(df[top_params[1]].min(), df[top_params[1]].max(), 50)
    P1, P2 = np.meshgrid(param1_range, param2_range)
    
    # Use real evaluations to simulate GP behavior
    real_evals = df[df['real_eval'] == True]
    
    if len(real_evals) < 10:
        print("Not enough real evaluations for GP uncertainty visualization")
        return
    
    # Function to estimate mean and uncertainty using kernel density estimation
    def estimate_gp(data, param1, param2, value_col):
        # Extract data points
        X = data[[param1, param2]].values
        y = data[value_col].values
        
        # Create grid points
        grid_points = np.vstack([P1.ravel(), P2.ravel()]).T
        
        # Calculate distance of each grid point to all data points
        mean_predictions = np.zeros(len(grid_points))
        uncertainty = np.zeros(len(grid_points))
        
        for i, point in enumerate(grid_points):
            # Calculate weighted average based on distance
            distances = np.sqrt(np.sum((X - point)**2, axis=1))
            weights = np.exp(-0.5 * distances)
            weights = weights / np.sum(weights)
            
            # Weighted mean
            mean_predictions[i] = np.sum(weights * y)
            
            # Weighted variance (uncertainty)
            uncertainty[i] = np.sum(weights * (y - mean_predictions[i])**2)
        
        # Reshape to grid
        mean_grid = mean_predictions.reshape(P1.shape)
        uncertainty_grid = uncertainty.reshape(P1.shape)
        
        return mean_grid, uncertainty_grid
    
    # Create visualizations for different generations
    generations = sorted(df['gen'].unique())
    early_gen = generations[min(5, len(generations)-1)]  # Gen 5 or earliest available
    mid_gen = generations[min(len(generations)//2, len(generations)-1)]  # Mid generation
    late_gen = generations[-1]  # Latest generation
    
    vis_generations = [early_gen, mid_gen, late_gen]
    
    for metric in ['fitness_psnr', 'fitness_time', 'fitness_memory']:
        fig, axes = plt.subplots(len(vis_generations), 2, figsize=(16, 4*len(vis_generations)))
        
        for i, gen in enumerate(vis_generations):
            # Get data up to this generation
            gen_data = real_evals[real_evals['gen'] <= gen]
            
            if len(gen_data) < 5:  # Skip if not enough data
                continue
                
            # Estimate GP mean and uncertainty
            mean_grid, uncertainty_grid = estimate_gp(gen_data, top_params[0], top_params[1], metric)
            
            # Plot mean prediction
            im1 = axes[i, 0].contourf(P1, P2, mean_grid, 50, cmap='viridis')
            axes[i, 0].scatter(gen_data[top_params[0]], gen_data[top_params[1]], 
                             c='red', edgecolor='black', s=50)
            axes[i, 0].set_xlabel(top_params[0])
            axes[i, 0].set_ylabel(top_params[1])
            axes[i, 0].set_title(f'Generation {gen}: Mean {metric.split("_")[1].upper()} Prediction')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Plot uncertainty
            im2 = axes[i, 1].contourf(P1, P2, uncertainty_grid, 50, cmap='plasma')
            axes[i, 1].scatter(gen_data[top_params[0]], gen_data[top_params[1]], 
                             c='red', edgecolor='black', s=50)
            axes[i, 1].set_xlabel(top_params[0])
            axes[i, 1].set_ylabel(top_params[1])
            axes[i, 1].set_title(f'Generation {gen}: {metric.split("_")[1].upper()} Uncertainty')
            plt.colorbar(im2, ax=axes[i, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"gp_uncertainty_{metric.split('_')[1]}.png"))
        plt.close(fig)
    
    # Create 1D slice visualization for one parameter
    top_param = top_params[0]
    param_range = np.linspace(df[top_param].min(), df[top_param].max(), 100)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    metrics = ['fitness_psnr', 'fitness_time', 'fitness_memory']
    titles = ['PSNR', 'Training Time', 'Memory Usage']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Get latest data
        latest_data = real_evals[real_evals['gen'] <= late_gen]
        
        # Calculate mean and std for each parameter value
        mean_vals = []
        std_vals = []
        
        for val in param_range:
            # Find nearest neighbors to this parameter value
            distances = np.abs(latest_data[top_param] - val)
            nearest = latest_data.iloc[np.argsort(distances)[:10]]
            
            if len(nearest) > 0:
                mean_vals.append(nearest[metric].mean())
                std_vals.append(nearest[metric].std())
            else:
                mean_vals.append(0)
                std_vals.append(0)
        
        # Convert to numpy arrays
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)
        
        # Plot mean and uncertainty
        axes[i].plot(param_range, mean_vals, 'b-', label='Mean Prediction')
        axes[i].fill_between(param_range, 
                            mean_vals - 2*std_vals, 
                            mean_vals + 2*std_vals, 
                            alpha=0.3, color='blue',
                            label='±2σ (95% Confidence)')
        
        # Plot actual data points
        axes[i].scatter(latest_data[top_param], latest_data[metric], 
                      c='red', edgecolor='black', s=50, alpha=0.7,
                      label='Actual Data')
        
        axes[i].set_xlabel(top_param)
        axes[i].set_ylabel(title)
        axes[i].set_title(f'1D GP Visualization: {title} vs {top_param}')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gp_1d_visualization.png"))
    plt.close(fig)

# 6. PARAMETER IMPORTANCE ANALYSIS
def analyze_parameter_importance(df):
    """Analyze and visualize which parameters have the most impact on fitness"""
    # Use real evaluations for more reliable analysis
    real_evals = df[df['real_eval'] == True]
    
    if len(real_evals) < 10:
        print("Not enough real evaluations for parameter importance analysis")
        return
        
    # Compute feature importance using permutation importance approach
    metrics = ['fitness_psnr', 'fitness_time', 'fitness_memory']
    importances = {metric: {} for metric in metrics}
    
    for metric in metrics:
        baseline = real_evals[metric].std()  # Baseline variation
        
        for param in param_cols:
            # Create a copy of the data with the parameter permuted
            shuffled_data = real_evals.copy()
            shuffled_data[param] = np.random.permutation(shuffled_data[param].values)
            
            # Calculate how much variation increases when this parameter is randomized
            permuted_std = shuffled_data.groupby(param)[metric].std().mean()
            
            # Higher importance means more variation when parameter is randomized
            # (i.e., parameter has more influence on outcome)
            importance = max(0, (permuted_std - baseline) / baseline)
            importances[metric][param] = importance
    
    # Normalize importance scores
    for metric in metrics:
        total = sum(importances[metric].values())
        if total > 0:
            for param in importances[metric]:
                importances[metric][param] /= total
    
    # Create bar plots for parameter importance
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics, 1):
        # Sort parameters by importance
        sorted_params = sorted(importances[metric].items(), key=lambda x: x[1], reverse=True)
        params = [x[0] for x in sorted_params]
        values = [x[1] for x in sorted_params]
        
        plt.subplot(3, 1, i)
        bars = plt.barh(params, values, color=plt.cm.viridis(np.linspace(0, 1, len(params))))
        
        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{width:.3f}', va='center')
        
        plt.xlabel('Relative Importance')
        plt.title(f'Parameter Importance for {metric.split("_")[1].upper()}')
        plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, "parameter_importance.png"))
    plt.close()
    
    # Create partial dependence plots for top parameters
    top_params = {}
    for metric in metrics:
        sorted_params = sorted(importances[metric].items(), key=lambda x: x[1], reverse=True)
        top_params[metric] = [x[0] for x in sorted_params[:3]]  # Top 3 parameters
    
    for metric in metrics:
        if len(top_params[metric]) == 0:
            continue
            
        fig, axes = plt.subplots(len(top_params[metric]), 1, figsize=(10, 4*len(top_params[metric])))
        if len(top_params[metric]) == 1:
            axes = [axes]
            
        for i, param in enumerate(top_params[metric]):
            # Create bins for parameter values
            param_values = np.linspace(real_evals[param].min(), real_evals[param].max(), 10)
            bin_means = []
            bin_stds = []
            bin_counts = []
            
            for j in range(len(param_values)-1):
                lower = param_values[j]
                upper = param_values[j+1]
                
                # Get data points in this bin
                mask = (real_evals[param] >= lower) & (real_evals[param] < upper)
                bin_data = real_evals[mask]
                
                if len(bin_data) > 0:
                    bin_means.append(bin_data[metric].mean())
                    bin_stds.append(bin_data[metric].std())
                    bin_counts.append(len(bin_data))
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)
                    bin_counts.append(0)
            
            # Convert to numpy arrays and handle NaN values
            bin_centers = (param_values[:-1] + param_values[1:]) / 2
            bin_means = np.array(bin_means)
            bin_stds = np.array(bin_stds)
            
            # Create scatter plot with size proportional to number of points
            axes[i].scatter(bin_centers, bin_means, 
                           s=[max(20, 100*(count/max(bin_counts))) for count in bin_counts],
                           alpha=0.7)
            
            # Plot error bars
            axes[i].errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none', alpha=0.5)
            
            # Plot smoothed line
            valid_indices = ~np.isnan(bin_means)
            if np.sum(valid_indices) > 1:
                from scipy.interpolate import make_interp_spline
                try:
                    spl = make_interp_spline(bin_centers[valid_indices], bin_means[valid_indices], k=min(3, sum(valid_indices)-1))
                    x_smooth = np.linspace(bin_centers[valid_indices].min(), bin_centers[valid_indices].max(), 100)
                    y_smooth = spl(x_smooth)
                    axes[i].plot(x_smooth, y_smooth, 'r-', alpha=0.7)
                except:
                    # Fall back to simple line if spline fails
                    axes[i].plot(bin_centers[valid_indices], bin_means[valid_indices], 'r-', alpha=0.7)
            
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric.split('_')[1].upper())
            axes[i].set_title(f'Partial Dependence: {metric.split("_")[1].upper()} vs. {param}')
            axes[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"partial_dependence_{metric.split('_')[1]}.png"))
        plt.close(fig)

# 7. PARAMETER DISTRIBUTION OVER GENERATIONS
def analyze_parameter_distributions(df):
    """Analyze how parameter distributions evolve over generations"""
    generations = sorted(df['gen'].unique())
    
    # Skip if too few generations
    if len(generations) < 2:
        print("Not enough generations for parameter distribution analysis")
        return
    
    # Select a subset of generations if there are too many
    if len(generations) > 8:
        step = len(generations) // 8
        selected_gens = generations[::step]
        if generations[-1] not in selected_gens:
            selected_gens.append(generations[-1])
    else:
        selected_gens = generations
    
    # 7.1. Violin plots for parameter distributions
    for param in param_cols:
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='gen', y=param, data=df[df['gen'].isin(selected_gens)])
        plt.title(f'Distribution of {param} Across Generations')
        plt.xlabel('Generation')
        plt.ylabel(param)
        
        # Rotate x-axis labels if needed
        if len(selected_gens) > 6:
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"param_dist_{param}.png"))
        plt.close()
    
    # 7.2. KDE plots for parameter evolution
    # Select a subset of important parameters
    if len(param_cols) > 6:
        # Use importance analysis from Section 6 to select parameters
        real_evals = df[df['real_eval'] == True]
        importance_scores = {}
        
        for param in param_cols:
            # Simple importance measure: variance of output when grouping by parameter
            importance = real_evals.groupby(pd.qcut(real_evals[param], 5, duplicates='drop'))['fitness_psnr'].std().mean()
            importance_scores[param] = importance
        
        # Sort parameters by importance
        sorted_params = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        selected_params = [x[0] for x in sorted_params[:6]]  # Top 6 parameters
    else:
        selected_params = param_cols
    
    # Create ridge plots (stacked KDE plots)
    for param in selected_params:
        plt.figure(figsize=(12, 8))
        
        # Define colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_gens)))
        
        # Create KDE plots
        for i, gen in enumerate(selected_gens):
            gen_data = df[df['gen'] == gen][param].dropna()
            
            # Skip if not enough data
            if len(gen_data) < 3:
                continue
                
            # Calculate KDE
            kde = gaussian_kde(gen_data)
            x_vals = np.linspace(df[param].min(), df[param].max(), 200)
            y_vals = kde(x_vals)
            
            # Normalize and offset
            y_vals = y_vals / y_vals.max() * 0.8  # Scale height
            offset = i  # Offset for stacking
            
            plt.fill_between(x_vals, offset, y_vals + offset, alpha=0.8, color=colors[i])
            plt.plot(x_vals, y_vals + offset, color='black', alpha=0.5)
            
            # Add generation label
            plt.text(df[param].min(), offset + 0.4, f'Gen {gen}', fontweight='bold', ha='left')
        
        plt.yticks([])  # Hide y-axis ticks
        plt.xlabel(param)
        plt.title(f'Evolution of {param} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"param_evolution_{param}.png"))
        plt.close()

# 8. TOP INDIVIDUALS TABLE
def create_top_individuals_table(df):
    """Create a table of top-performing individuals"""
    # Identify Pareto-optimal individuals using real evaluations
    real_evals = df[df['real_eval'] == True]
    
    if len(real_evals) == 0:
        print("No real evaluations for top individuals table")
        return
    
    # Determine Pareto-optimal points
    is_pareto = np.ones(len(real_evals), dtype=bool)
    for i, (psnr_i, time_i, mem_i) in enumerate(zip(
        real_evals['fitness_psnr'], real_evals['fitness_time'], real_evals['fitness_memory'])):
        for psnr_j, time_j, mem_j in zip(
            real_evals['fitness_psnr'], real_evals['fitness_time'], real_evals['fitness_memory']):
            # Check if point j dominates point i
            if ((psnr_j > psnr_i and time_j <= time_i and mem_j <= mem_i) or
                (psnr_j >= psnr_i and time_j < time_i and mem_j <= mem_i) or
                (psnr_j >= psnr_i and time_j <= time_i and mem_j < mem_i)):
                is_pareto[i] = False
                break
    
    # Get Pareto-optimal individuals
    pareto_individuals = real_evals[is_pareto].copy()
    
    # Also get top individuals for each metric
    top_psnr = real_evals.nlargest(3, 'fitness_psnr').copy()
    top_time = real_evals.nsmallest(3, 'fitness_time').copy()
    top_memory = real_evals.nsmallest(3, 'fitness_memory').copy()
    
    # Combine all top individuals and remove duplicates
    all_top = pd.concat([pareto_individuals, top_psnr, top_time, top_memory])
    all_top = all_top.drop_duplicates(subset=['gen', 'index'])
    
    # Sort by PSNR (descending)
    all_top = all_top.sort_values('fitness_psnr', ascending=False)
    
    # Create HTML table with styling
    html_file = os.path.join(plots_dir, "top_individuals.html")
    
    # Identify columns to include
    display_cols = ['gen', 'fitness_psnr', 'fitness_time', 'fitness_memory'] + param_cols
    
    # Create styled HTML
    with open(html_file, 'w') as f:
        f.write('<html>\n<head>\n')
        f.write('<style>\n')
        f.write('table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }\n')
        f.write('th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }\n')
        f.write('th { background-color: #4CAF50; color: white; }\n')
        f.write('tr:nth-child(even) { background-color: #f2f2f2; }\n')
        f.write('tr:hover { background-color: #ddd; }\n')
        f.write('caption { font-size: 1.5em; margin-bottom: 10px; }\n')
        f.write('.pareto { background-color: #ffffcc; }\n')  # Highlight Pareto points
        f.write('.top-psnr { color: #4CAF50; font-weight: bold; }\n')
        f.write('.top-time { color: #2196F3; font-weight: bold; }\n')
        f.write('.top-memory { color: #f44336; font-weight: bold; }\n')
        f.write('</style>\n</head>\n<body>\n')
        
        f.write('<h1>Top Individuals from Hyperparameter Optimization</h1>\n')
        f.write('<table>\n')
        f.write('<caption>Top Performing and Pareto-Optimal Individuals</caption>\n')
        
        # Write header
        f.write('<tr>\n')
        for col in display_cols:
            f.write(f'<th>{col}</th>\n')
        f.write('</tr>\n')
        
        # Write rows
        for idx, row in all_top.iterrows():
            # Determine if this is a Pareto point
            is_pareto_point = idx in pareto_individuals.index
            row_class = 'pareto' if is_pareto_point else ''
            
            f.write(f'<tr class="{row_class}">\n')
            
            for col in display_cols:
                cell_value = row[col]
                
                # Apply special styling for fitness metrics
                cell_class = ''
                if col == 'fitness_psnr' and idx in top_psnr.index:
                    cell_class = 'top-psnr'
                elif col == 'fitness_time' and idx in top_time.index:
                    cell_class = 'top-time'
                elif col == 'fitness_memory' and idx in top_memory.index:
                    cell_class = 'top-memory'
                
                # Format numeric values
                if isinstance(cell_value, (int, float)):
                    if col.startswith('fitness_'):
                        formatted_val = f'{cell_value:.3f}'
                    else:
                        formatted_val = f'{cell_value:.4f}' if cell_value < 0.01 else f'{cell_value:.3f}'
                else:
                    formatted_val = str(cell_value)
                
                f.write(f'<td class="{cell_class}">{formatted_val}</td>\n')
            
            f.write('</tr>\n')
        
        f.write('</table>\n')
        f.write('</body>\n</html>')
    
    print(f"Top individuals table saved as HTML: {html_file}")
    
    # Also create a CSV version for easier data processing
    csv_file = os.path.join(plots_dir, "top_individuals.csv")
    all_top[display_cols].to_csv(csv_file, index=False)
    print(f"Top individuals also saved as CSV: {csv_file}")

# Main function to run all visualizations
def run_all_visualizations(df):
    """Run all visualization functions"""
    print("Creating Pareto Front Visualizations...")
    plot_pareto_front(df)
    
    print("Creating Pareto Front Animation...")
    animate_pareto_front(df)
    
    print("Creating Fitness Evolution Plots...")
    plot_fitness_evolution(df)
    
    print("Analyzing Population Diversity...")
    plot_population_diversity(df)
    
    print("Analyzing Surrogate Model Accuracy...")
    plot_surrogate_accuracy(df)
    
    print("Visualizing Gaussian Process Uncertainty...")
    visualize_gp_uncertainty(df)
    
    print("Analyzing Parameter Importance...")
    analyze_parameter_importance(df)
    
    print("Analyzing Parameter Distributions...")
    analyze_parameter_distributions(df)
    
    print("Creating Top Individuals Table...")
    create_top_individuals_table(df)
    
    print("All visualizations complete! Output saved to:", plots_dir)

# Execute all visualizations
if __name__ == "__main__":
    # Print information about the dataset
    print("DataFrame info:")
    print(f"Total individuals: {len(df)}")
    print(f"Number of generations: {df['gen'].nunique()}")
    print(f"Hyperparameters: {param_cols}")
    
    # Run all visualizations
    run_all_visualizations(df)
