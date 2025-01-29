import os
import numpy as np
from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv
import matplotlib.pyplot as plt



tools_of_interest = ["bw_Martillo01.jpg", "empty.png", "bw_Lapicera01.png", "bw_destornillador01.jpg", "bw_tornillo01.jpg"]

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
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest=[tool_name])

    hidden_layers=[32,32] 

    agent = Agent(env=env, hidden_layers=hidden_layers, noise=0.0) 

    agent.load_models()

    episodes = 500

    for i in range(episodes):
        observation = env.reset()
        tool = agent.observer(observation[0]).cpu().detach().numpy() # Takes the image and outputs a tool value
        observation = np.array([tool.item(), observation[1]]) # [tool, f_idx]
        done = False
        score = 0
        right_comb_ctr = 0
        while not done:
            action = agent.choose_action(observation, validation=True) 
            next_observation, reward, done, info = env.step(action)
            next_observation = np.array([tool.item(), next_observation[1]]) # [tool, f_idx]
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


# Number of rows and columns for subplots
cols = 5
rows = 1

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
plt.tight_layout()
plt.show()

print("weird combinations:", weird_combinations)