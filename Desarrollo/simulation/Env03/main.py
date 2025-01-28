import os
from torch.utils.tensorboard import SummaryWriter
from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv
import numpy as np

if __name__ == '__main__':

    if not os.path.exists("Desarrollo/simulation/Env03/tmp/td3"):
        os.makedirs("Desarrollo/simulation/Env03/tmp/td3")

    #images_of_interest = ["bw_Martillo01.jpg", "empty.png", "bw_Lapicera01.png", "bw_destornillador01.jpg", "bw_tornillo01.jpg"]
    images_of_interest = "all"
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest=images_of_interest)

    load_models = False
    actor_learning_rate = 0.003 #0.001    1.0
    critic_learning_rate = 0.0008 #0.001   0.001
    batch_size = 64 #128

    hidden_layers=[32,16] #256
    warmup = 1200 * 5
    episodes = 10000 #10000
    env.reward_weights["reward_alpha"] = 1

    max_size = 100000  # Adjust this value based on memory capacity

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                  tau=0.005, env=env, n_actions=env.n_fingers, hidden_layers=hidden_layers, 
                  batch_size=batch_size, warmup=warmup, max_size=max_size) 
    agent.train()

    writer = SummaryWriter("Desarrollo/simulation/Env03/logs")

    for experiment in range(0, 1):
        best_score = 0

        directory_path = "Desarrollo/simulation/Env03/logs_txt/"
        version = "e4"
        file_name = f"experiment_log_{experiment}_{version}.txt"

        while os.path.exists(directory_path + file_name):
            experiment += 1
            print("Experiment already exists. Trying with experiment number: ", experiment)
            file_name = f"experiment_log_{experiment}_{version}.txt"
        
        if load_models:
            agent.load_models()
            experiment -=1
            file_name = f"experiment_log_{experiment}_{version}.txt"
            
        print("Starting:",file_name)

        episode_identifier = f"{experiment} - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} hidden_layers={hidden_layers} warmup={warmup} reward_alpha={env.reward_weights['reward_alpha']} batch_size={batch_size}  _{version}"

        log_file = open(directory_path + file_name, "a")

        for i in range(episodes):
            observation = env.reset()
            tool = agent.observer(observation[0]).cpu().detach().numpy() # Takes the image and outputs a tool value
            observation = np.array([tool.item(), observation[1]]) # [tool, f_idx]
            done = False
            score = 0
            while not done:
                action = agent.choose_action(observation) 
                next_observation, reward, done, info = env.step(action)
                next_observation = np.array([tool.item(), next_observation[1]]) # [tool, f_idx]
                score += reward
                agent.remember(observation, action, reward, next_observation, done)
                agent.learn()
                observation = next_observation

            writer.add_scalar(f"Score - {episode_identifier}", scalar_value=score, global_step=i)

            # Log the information to the text file
            log_file.write(f"Episode: {i}; Score: {score}; Action: {env.complete_action()}\n")
            print(f"Episode: {i}; Score: {score}; Tool: {tool}; Action: {env.complete_action()}")

            if i % 10 == 0:
                agent.save_models()

        # Close the log file
        log_file.close()

        """# Hiperparámetros a buscar:
        actor_learning_rate *= 0.8
        critic_learning_rate *= 0.8
        
        if experiment % 2 == 0:
            env.reward_weights["reward_alpha"] *= 2
            actor_learning_rate = 0.001 # quizá probar también con 0.01
            critic_learning_rate = 0.001

        if experiment % 6 == 0:
            #conv_channels=[16, 32, 64], hidden_size=256
            actor_learning_rate = 0.001 # quizá probar también con 0.01
            critic_learning_rate = 0.001
            env.reward_weights["reward_alpha"] = 1"""

        
        
        
