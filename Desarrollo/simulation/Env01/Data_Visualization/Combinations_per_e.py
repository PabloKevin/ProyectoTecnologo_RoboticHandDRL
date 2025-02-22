import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Directory containing the log files
directory_path = "/home/pablo_kevin/ProyectoTecnologo_RoboticHandDRL/Desarrollo/simulation/Env01/logs_txt/"

# Get the list of files in the directory
log_files = [file for file in os.listdir(directory_path) if file.endswith("e2.txt")]

# Function to parse the log file
def parse_log(file_path):
    # Create lists to store episode details
    episodes, scores, actions = [], [], []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.split(";")
            if len(parts) < 3:  # Ensure the line has the expected format
                continue
            # Extract Episode
            episode = int(parts[0].split(":")[1].strip())
            # Extract Score
            score = float(parts[1].split(":")[1].strip())
            # Extract Actions
            action_str = parts[2].split(":")[1].replace('[', '').replace(']', '')
            action_list = [int(float(x)) for x in action_str.split(" ") if (x != '' and x != '\n')]
            episodes.append(episode)
            scores.append(score)
            actions.append(action_list)
    return pd.DataFrame({"Episode": episodes, "Score": scores, "Actions": actions})

# Action combinations to check
combinations = {
    "Combination 1": [2, 1, 2, 2, 2],  # Thumb closed, index half, others closed
    "Combination 2": [2, 1, 1, 2, 2],  # Thumb closed, index and middle half, others open
    "Combination 3": [1, 1, 1, 1, 1],  # All fingers half closed
    "Combination 4": [0, 0, 0, 0, 0]   # All fingers opened
}

# Maximum files per figure
files_per_figure = 12

# Create multiple figures if needed
for fig_idx in range(math.ceil(len(log_files) / files_per_figure)):
    # Determine the files for the current figure
    start_idx = fig_idx * files_per_figure
    end_idx = start_idx + files_per_figure
    current_files = log_files[start_idx:end_idx]
    
    # Number of rows and columns for subplots
    cols = 6
    rows = math.ceil(len(current_files) / cols)
    
    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(17, 10), constrained_layout=False)
    axes = axes.flatten() if rows > 1 else [axes]
    
    # Loop through the files for the current figure
    for idx, file_name in enumerate(current_files):
        log_file_path = os.path.join(directory_path, file_name)
        
        # Parse the log
        df = parse_log(log_file_path)
        
        # Calculate frequencies of combinations
        combination_counts = {name: 0 for name in combinations.keys()}
        for action in df["Actions"]:
            for name, combination in combinations.items():
                if action == combination:
                    combination_counts[name] += 1
        
        # Plot on the corresponding subplot
        ax = axes[idx]
        ax.bar(combination_counts.keys(), combination_counts.values(), color="orange", edgecolor="black")
        ax.set_xlabel("Action Combinations", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.set_title(f"File: {file_name}\n(Total Episodes: {len(df)})", fontsize=10)
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # Hide unused subplots
    for idx in range(len(current_files), len(axes)):
        axes[idx].axis("off")
    
    # Add overall title for the current figure
    fig.suptitle(f"Frequency of Combinations of Interest (Figure {fig_idx + 1})", fontsize=16)
    plt.tight_layout()
    plt.show()
