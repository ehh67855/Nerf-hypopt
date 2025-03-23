import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# create a directory to save plots if it doesn't exist
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# load optimization logs from json files
log_dir = "logs/optimization"  # directory containing JSON files
all_data = []
for file in sorted(os.listdir(log_dir)):
    if file.startswith("gen_") and file.endswith(".json"):
        with open(os.path.join(log_dir, file)) as f:
            gen_data = json.load(f)
            for entry in gen_data:
                # flatten the params dictionary and combine it with other fields
                flattened_entry = {
                    **entry['params'],  # flatten params into top-level keys
                    'gen': entry['gen'],
                    'index': entry['index'],
                    'fitness_psnr': entry['fitness'][0],  # PSNR
                    'fitness_time': entry['fitness'][1],  # Training Time
                    'fitness_memory': entry['fitness'][2],  # Memory Usage
                    'real_eval': entry['real_eval']
                }
                all_data.append(flattened_entry)

# convert data to dataframe for easier manipulation
df = pd.DataFrame(all_data)

# inspect dataframe structure for debugging purposes
print("DataFrame Columns:", df.columns)
print(df.head())

# plot pareto front showing trade-offs between objectives
def plot_pareto_front(df):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df['fitness_psnr'], df['fitness_time'], c=df['fitness_memory'], cmap='viridis', s=50
    )
    plt.colorbar(scatter, label='Memory Usage')
    plt.xlabel('PSNR')
    plt.ylabel('Training Time')
    plt.title('Pareto Front: PSNR vs. Time vs. Memory')
    plt.savefig(os.path.join(plots_dir, "pareto_front.png"))  # save plot to file
    plt.show()

# animate pareto front evolution (assign animation to a persistent variable)
def animate_pareto_front():
    generations = df['gen'].unique()  # unique generations in the data
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter([], [], c=[], cmap='viridis', s=50)
    ax.set_xlim(df['fitness_psnr'].min() - 1, df['fitness_psnr'].max() + 1)
    ax.set_ylim(df['fitness_time'].min() - 1, df['fitness_time'].max() + 1)
    ax.set_xlabel('PSNR')
    ax.set_ylabel('Training Time')
    ax.set_title('Pareto Front Animation: PSNR vs. Time')

    cbar = plt.colorbar(sc)
    cbar.set_label('Memory Usage')

    def update(frame):
        gen_data = df[df['gen'] == generations[frame]]  # filter data for current generation
        sc.set_offsets(gen_data[['fitness_psnr', 'fitness_time']].values)  # update scatter points
        sc.set_array(gen_data['fitness_memory'])  # update color array
        ax.set_title(f'Pareto Front: Generation {generations[frame]}')  # update title
        return sc,

    anim = FuncAnimation(fig, update, frames=len(generations), interval=500, blit=False)
    
    anim.save(os.path.join(plots_dir, "pareto_front_animation.mp4"), fps=10)  # save animation to file
    print(f"Animation saved as pareto_front_animation.mp4")
    
    plt.show()

# plot fitness evolution over generations
def plot_fitness_evolution(df):
    generations = df['gen'].unique()
    best_psnr = [df[df['gen'] == gen]['fitness_psnr'].max() for gen in generations]
    avg_psnr = [df[df['gen'] == gen]['fitness_psnr'].mean() for gen in generations]
    worst_psnr = [df[df['gen'] == gen]['fitness_psnr'].min() for gen in generations]

    plt.figure(figsize=(8, 6))
    plt.plot(generations, best_psnr, label='Best PSNR', color='green')
    plt.plot(generations, avg_psnr, label='Average PSNR', color='blue')
    plt.plot(generations, worst_psnr, label='Worst PSNR', color='red')
    plt.fill_between(generations, worst_psnr, best_psnr, color='gray', alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('PSNR')
    plt.title('Fitness Evolution Over Generations')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "fitness_evolution.png"))  # save plot to file
    plt.show()

# run visualizations only if required columns exist
try:
    plot_pareto_front(df)  # plot pareto front and save it as an image
    plot_fitness_evolution(df)  # plot fitness evolution and save it as an image
    
    anim = animate_pareto_front()  # animate pareto front and save it as a video
    
except KeyError as e:
    print(f"Error: Missing column {e}")
