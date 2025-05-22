import os
from torch.utils.tensorboard import SummaryWriter
from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv
import numpy as np
import torch.nn.functional as F

if __name__ == '__main__':

    if not os.path.exists("Desarrollo/simulation/Env05/tmp/td3"):
        os.makedirs("Desarrollo/simulation/Env05/tmp/td3")

    #images_of_interest = ["bw_Martillo01.jpg", "empty.png", "bw_Lapicera01.png", "bw_destornillador01.jpg", "bw_tornillo01.jpg", "bw_lapicera02"]
    images_of_interest = "all"
    #images_of_interest = ["lapicera"]
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest=images_of_interest, dataset_name=["TrainSet_masks","TestSet_masks"])

    load_models = False
    actor_learning_rate = 0.001 #0.001    1.0
    critic_learning_rate = 0.0008 #0.001   0.001
    batch_size = 128 #128

    hidden_layers=[32,16,8,4] #256
    warmup = 1200 * 5
    episodes = 6000 #10000
    env.reward_weights["reward_alpha"] = 1

    max_size = 100000  # Adjust this value based on memory capacity

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                  tau=0.005, env=env, hidden_layers=hidden_layers, 
                  batch_size=batch_size, warmup=warmup, max_size=max_size) 
    agent.train()

    writer = SummaryWriter("Desarrollo/simulation/Env05/logs")

    seq_len = 5  # Set this at the top

    for experiment in range(0, 1):
        best_score = 0

        directory_path = "Desarrollo/simulation/Env05/logs_txt/"
        version = "e6"
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
            tool = observation[:-1]
            done = False
            score = 0

            obs_seq = [env.reset() for _ in range(seq_len)]  # Initialize with initial obs

            while not done:
                if load_models:
                    action = agent.choose_action(observation, validation=True) 
                else:
                    action = agent.choose_action(observation) 
                next_observation, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, next_observation, done)
                agent.learn()
                observation = next_observation

                agent.memory.store_transition(obs_seq[-1], action, reward, next_observation, done)
                obs_seq.append(next_observation)
                obs_seq = obs_seq[-seq_len:]  # Keep only the last seq_len observations

            writer.add_scalar(f"Score - {episode_identifier}", scalar_value=score, global_step=i)

            # Log the information to the text file
            log_file.write(f"Episode: {i}; Tool: {tool}; Score: {score}; Action: {env.complete_action()}\n")
            print(f"Episode: {i}; Score: {score}; Tool: {tool}; Action: {env.complete_action()}")

            if i % 10 == 0:
                agent.save_models()

            agent.actor_scheduler.step()
            agent.critic1_scheduler.step()
            agent.critic2_scheduler.step()
            agent.target_actor_scheduler.step()
            agent.target_critic1_scheduler.step()
            agent.target_critic2_scheduler.step()
        
        # Close the log file
        log_file.close()





