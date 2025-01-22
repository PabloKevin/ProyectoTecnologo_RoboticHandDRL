import time
import os
#import numpy as np
from torch.utils.tensorboard import SummaryWriter

#from networks import CriticNetwork,ActorNetwork
#from buffer import ReplayBuffer
from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv

if __name__ == '__main__':

    if not os.path.exists("Desarrollo/simulation/Env03/tmp/td3"):
        os.makedirs("Desarrollo/simulation/Env03/tmp/td3")

    # Create an instance of your custom environment
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=5)
# Ir probando con numeros más simples para que lleve menos tiempo. Dado que el problemas es más simple,
# usar menos neuronas, probablemente no necesite tantas imágenes para aprender. Quizá probar con 1 sola capa.
    load_models = False
    actor_learning_rate = 0.001 #0.001    1.0
    critic_learning_rate = 0.0008 #0.001   0.001
    batch_size = 64 #128

    hidden_layers=[128,64] #256
    warmup = 1200
    episodes = 7000 #10000
    env.reward_weights["reward_alpha"] = 1

    # Reduce the replay buffer size
    max_size = 10000  # Adjust this value based on your memory capacity

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.05, #tau=0.005
                  env=env, n_actions=env.n_fingers, n_choices_per_finger=env.n_choices_per_finger, hidden_layers=hidden_layers, 
                  batch_size=batch_size, warmup=warmup, max_size=max_size) 

    #print("n_actions: ", agent.n_actions)
    writer = SummaryWriter("Desarrollo/simulation/Env03/logs")

    
    
    for experiment in range(0, 1):
        best_score = 0

        directory_path = "Desarrollo/simulation/Env03/logs_txt/"
        version = "e0"
        file_name = f"experiment_log_{experiment}_{version}.txt"
        file_name_probs = f"probabilities_log_{experiment}_{version}.txt"

        while os.path.exists(directory_path + file_name):
            experiment += 1
            print("Experiment already exists. Trying with experiment number: ", experiment)
            file_name = f"experiment_log_{experiment}_{version}.txt"
            file_name_probs = f"probabilities_log_{experiment}_{version}.txt"
        
        if load_models:
            agent.load_models()
            experiment -=1
            file_name = f"experiment_log_{experiment}_{version}.txt"
            file_name_probs = f"probabilities_log_{experiment}_{version}.txt"
            
        print("Starting:",file_name)

        episode_identifier = f"{experiment} - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} hidden_layers={hidden_layers} warmup={warmup} reward_alpha={env.reward_weights['reward_alpha']} batch_size={batch_size}  _{version}"

        # Open a log file in append mode
        log_file = open(directory_path + file_name, "a")
        log_file_probs = open(directory_path + file_name_probs, "a")
            
        #agent.load_models()

        for i in range(episodes):
            observation = env.reset()
            observation = agent.observer(observation) # Takes the image and outputs a tool value
            score = 0
            action_probs = agent.choose_action(observation)
            action = agent.env.probs2actions(action_probs)
            next_observation, reward, info = env.step(action)
            score += reward
            agent.remember(observation, action_probs, reward)
            agent.learn()

            writer.add_scalar(f"Score - {episode_identifier}", scalar_value=score, global_step=i)

            # Log the information to the text file
            log_file.write(f"Episode: {i}; Score: {score}; Action: {action}\n")
            log_file_probs.write(f"probabilities: {action_probs}\n")
            print(f"Episode: {i}; Score: {score}; Action: {action}")

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

        
        
        
