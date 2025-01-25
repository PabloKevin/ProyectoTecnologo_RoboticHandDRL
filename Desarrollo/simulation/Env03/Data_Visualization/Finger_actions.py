import pandas as pd
import matplotlib.pyplot as plt

# Function to parse the log file
def parse_log(file_path):
    # Create lists to store episode details
    episodes, scores, actions = [], [], []
    
    with open(file_path, "r") as file:
        for line in file:     
            # Split the line by semicolon
            parts = line.split(";")
            if len(parts) < 3:  # Ensure the line has the expected format
                continue
            
            # Extract Episode
            episode = int(parts[0].split(":")[1].strip())
            
            # Extract Score
            score = float(parts[1].split(":")[1].replace('[', '').replace(']', '').strip())
            
            # Extract Actions
            action_raw_str = parts[2].split(":")[1].split("],")[0].replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            action_str = parts[2].split(":")[1].split("],")[1].replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            action_list = [float(x) for x in action_str.split(",")]
            action_raw_list = [float(x) for x in action_raw_str.split(",")]
            
            # Append to lists
            episodes.append(episode)
            scores.append(score)
            actions.append(action_list)
    
    # Create a DataFrame
    df = pd.DataFrame({
        "Episode": episodes,
        "Score": scores,
        "Actions": actions
    })
    return df

# Path to your episode log file
directory_path = "/home/pablo_kevin/ProyectoTecnologo_RoboticHandDRL/Desarrollo/simulation/Env03/logs_txt/"
file_name = "experiment_log_0_e4.txt"
log_file_path = directory_path + file_name

# Parse the log and load it into a DataFrame
df = parse_log(log_file_path)

# Separate warmup and training
warmup_cutoff = 1200
actions_all = [list(actions) for actions in df["Actions"]]
actions_warmup = actions_all[:warmup_cutoff]
actions_training = actions_all[warmup_cutoff:]

# Colors for each row
row_colors = ['skyblue', 'lightgreen', 'lightcoral']
# Titles for each row
row_titles = ["All Frequency", "Warmup Frequency", "Training Frequency"]

# Create subplots
fig, axes = plt.subplots(3, 5, figsize=(22, 12), constrained_layout=False)

# Iterate through fingers (columns)
for finger_idx in range(5):
    for row_idx, (title, actions_subset, color) in enumerate(zip(row_titles, [actions_all, actions_warmup, actions_training], row_colors)):
        finger_actions = [actions[finger_idx] for actions in actions_subset]
        axes[row_idx, finger_idx].hist(
            finger_actions,
            bins=range(int(min(finger_actions)), int(max(finger_actions)) + 2),
            align='left',
            edgecolor='black',
            color=color
        )
        if row_idx == 0:
            axes[row_idx, finger_idx].set_title(f"Finger {finger_idx + 1}")
        if finger_idx == 0:
            # Add row titles
            axes[row_idx, finger_idx].annotate(
                title,
                xy=(-0.7, 0.5),
                xycoords='axes fraction',
                fontsize=16,
                ha='center',
                va='center',
                rotation=0
            )
        axes[row_idx, finger_idx].set_xlabel("Action")

# Adjust padding to create more space at the top and right
fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1, hspace=0.5, wspace=0.4)

# Add overall title
fig.suptitle(f"Action Distribution per Finger of {file_name[:-4]}", fontsize=18)
plt.show()


# Calculate frequencies of combinations
# Action combinations to check
combinations = {
    "Combination 1": [2, 1, 2, 2, 2],  # Thumb closed, index half, others closed
    "Combination 2": [2, 1, 1, 2, 2],  # Thumb closed, index and middle half, others open
    "Combination 3": [1, 1, 1, 1, 1],     # All fingers half closed
    "Combination 4": [0, 0, 0, 0, 0]           # All fingers opened
}

combination_counts = {name: 0 for name in combinations.keys()}
for action in df["Actions"]:
    for name, combination in combinations.items():
        if action == combination:
            combination_counts[name] += 1

# Create the histogram
plt.bar(combination_counts.keys(), combination_counts.values(), color="orange", edgecolor="black")
plt.xlabel("Action Combinations")
plt.ylabel("Frequency")
plt.title(f"Frequency of Specific Action Combinations, total episodes: {len(df)}")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()