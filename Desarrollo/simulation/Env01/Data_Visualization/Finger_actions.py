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
            score = float(parts[1].split(":")[1].strip())
            
            # Extract Actions
            action_str = parts[2].split(":")[1].replace('[', '').replace(']', '')
            action_list = [float(x) for x in action_str.split(" ") if (x!='' and x!='\n')]
            
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
log_file_path = "/home/pablo_kevin/ProyectoTecnologo_RoboticHandDRL/Desarrollo/simulation/Env01/logs_txt/episode_log_0.txt"

# Parse the log and load it into a DataFrame
df = parse_log(log_file_path)

# Extract the first finger's actions
first_finger_actions = [actions[0] for actions in df["Actions"]]
#print("min:",min(first_finger_actions))
# Plot a histogram of the first finger's actions
plt.hist(first_finger_actions, bins=range(int(min(first_finger_actions)), int(max(first_finger_actions)) + 2), align='left', edgecolor='black')
plt.xlabel("Action of First Finger")
plt.ylabel("Frequency")
plt.title("Histogram of First Finger Actions")
plt.show()

plt.hist(first_finger_actions[:1000], bins=range(int(min(first_finger_actions)), int(max(first_finger_actions)) + 2), align='left', edgecolor='black')
plt.xlabel("Action of First Finger")
plt.ylabel("Frequency")
plt.title("Histogram of First Finger Actions, Warm Up")
plt.show()

plt.hist(first_finger_actions[1000:], bins=range(int(min(first_finger_actions)), int(max(first_finger_actions)) + 2), align='left', edgecolor='black')
plt.xlabel("Action of First Finger")
plt.ylabel("Frequency")
plt.title("Histogram of First Finger Actions, Training")
plt.show()
#print(df.info)
