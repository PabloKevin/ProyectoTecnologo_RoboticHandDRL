import os
import numpy as np
from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv
import matplotlib.pyplot as plt
from collections import Counter
import math


tools_of_interest = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]
#tools_of_interest = ["empty", "tuerca"]

# Action combinations to check
combinations = {
    "Combination 1": [2, 1, 2, 2, 2],  # Thumb closed, index half, others closed
    "Combination 2": [2, 1, 1, 2, 2],  # Thumb closed, index and middle half, others open
    "Combination 3": [1, 1, 1, 1, 1],     # All fingers half closed
    "Combination 4": [0, 0, 0, 0, 0],           # All fingers opened
    "Weird Combination": None,              # Any other combination
}

combination_counts_per_tool = [{name: 0 for name in combinations.keys()} for _ in range(len(tools_of_interest))]
weird_combinations = {name: [] for name in tools_of_interest}

for j, tool_name in enumerate(tools_of_interest):
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest=[tool_name], dataset_dir="Desarrollo/simulation/Env03/DataSets/TestSet_masks")

    hidden_layers=[256,128] 

    agent = Agent(env=env, hidden_layers=hidden_layers, noise=0.0) 

    agent.load_models()

    episodes = 400

    for i in range(episodes):
        observation = env.reset()
        tool, f_idx = observation
        done = False
        score = 0
        right_comb_ctr = 0
        while not done:
            action = agent.choose_action(observation, validation=True) 
            next_observation, reward, done, info = env.step(action)
            score += reward
            observation = next_observation

        weird_ctr_aux = 0
        for name, combination in combinations.items():
            if env.complete_action()[1] == combination:
                combination_counts_per_tool[j][name] += 1
            else:
                weird_ctr_aux += 1
        if weird_ctr_aux == 5:
            combination_counts_per_tool[j]["Weird Combination"] += 1
            weird_combinations[tool_name].append(env.complete_action()[1])

# A helper function to format weird combos
def format_weird_combos(tool_weird_list):
    """
    Converts the list of weird combos into a string like:
    "weird combos: [2,1,1,1,1] x 5, [2,1,1,1,2] x 1"
    or "No weird combos" if empty.
    """
    if not tool_weird_list:
        return "No weird combos"
    # Convert combos to tuples so they are hashable by Counter
    combos_as_tuples = [tuple(c) for c in tool_weird_list]
    combo_counter = Counter(combos_as_tuples)
    # Build each part of the string
    parts = []
    for combo, freq in combo_counter.items():
        parts.append(f"{list(combo)} x {freq}")
    return "weird combos: " + ", ".join(parts)

# Number of rows and columns for subplots
# Number of rows and columns for subplots
cols = 5
rows = math.ceil(len(tools_of_interest) / cols)

# Create the figure and axes
fig, axes = plt.subplots(rows, cols, figsize=(17, 10), constrained_layout=False)
axes = axes.flatten()  # Flatten so we can index easily

# Loop through the files for the current figure
for idx, tool in enumerate(tools_of_interest):
    ax = axes[idx]
    
    # Extract keys and values
    x_labels = list(combination_counts_per_tool[idx].keys())
    frequencies = list(combination_counts_per_tool[idx].values())
    
    # Generate a random color for each subplot (optional)
    random_color = np.random.rand(3,)
    
    # Plot the bars
    bar_container = ax.bar(
        x_labels, 
        frequencies, 
        color=random_color, 
        edgecolor="black"
    )
    
    # Add labels on top of the bars
    for bar in bar_container:
        bar_height = bar.get_height()
        # x-position is bar's center
        x_position = bar.get_x() + bar.get_width() / 2  
        # y-position is bar's top
        y_position = bar_height
        
        # Draw the text just above the bar
        ax.text(
            x_position,
            y_position,
            f"{bar_height}",  # format the number as needed
            ha='center',         # horizontally center the text
            va='bottom',         # put the text just above the bar
            fontsize=8,
            rotation=0,
            color='black'
        )
    
    # Format the weird combos text for the current tool
    weird_text = format_weird_combos(weird_combinations.get(tool, []))
    
    # Add the weird combo text *below* the x-axis. 
    # ax.transAxes: (0,0) is bottom-left of the entire Axes; (1,1) is top-right
    ax.text(
        0.5,
        -0.2,                 # negative moves the text below the x-axis
        weird_text,
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=8,
        wrap=True  # wrap text if it's too long
    )
    
    # Axes formatting
    ax.set_xlabel("Action Combinations", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.set_title(tool, fontsize=10)
    ax.tick_params(axis='x', rotation=15, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# Add overall title for the current figure
fig.suptitle(
    f"Frequency of Specific Action Combinations per tool\nTotal episodes per tool: {episodes}",
    fontsize=16
)
plt.xticks(rotation=15)
# Increase bottom margin to accommodate the weird combos text
plt.tight_layout()
#plt.subplots_adjust(bottom=0.5)
plt.show()

print("weird combinations:", weird_combinations)