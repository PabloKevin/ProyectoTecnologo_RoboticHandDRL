import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper

from networks import CriticNetwork,ActorNetwork
from buffer import ReplayBuffer
from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv

if __name__ == '__main__':

    if not os.path.exists("Desarrollo/simulation/Env01/tmp/td3"):
        os.makedirs("Desarrollo/simulation/Env01/tmp/td3")

    # Create an instance of your custom environment
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=5)
# Ir probando con numeros más simples para que lleve menos tiempo. Dado que el problemas es más simple,
# usar menos neuronas, probablemente no necesite tantas imágenes para aprender. Quizá probar con 1 sola capa.
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 64 #32/64 , the antecedente was 256/128
    layer2_size = 32
    warmup = 1000

    # Reduce the replay buffer size
    max_size = 10000  # Adjust this value based on your memory capacity

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.get_observation_space_shape(), env=env, n_actions=env.n_fingers,
                  layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, warmup=warmup,
                  max_size=max_size)  # Pass the max_size to the Agent

    #print("n_actions: ", agent.n_actions)
    writer = SummaryWriter("Desarrollo/simulation/Env01/logs")
    episodes = 7000 #10000 recomendados en el video
    
    
    for experiment in range(0,1):
        best_score = 0

        episode_identifier = f"{experiment} - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size} _ RoboticHand"

        #agent.load_models()

        # Open a log file in append mode
        log_file = open(f"Desarrollo/simulation/Env01/logs_txt/episode_log_{experiment}.txt", "a")

        for i in range(episodes):
            observation = env.reset()
            done = False
            score = 0

            while not done:
                action = agent.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, next_observation, done)
                agent.learn()
                observation = next_observation

            writer.add_scalar(f"Score - {episode_identifier}", scalar_value=score, global_step=i)

            # Log the information to the text file
            log_file.write(f"Episode: {i}; Score: {score}; Action: {action}\n")
            print(f"Episode: {i}; Score: {score}; Action: {action}")

            if i % 10 == 0:
                agent.save_models()

        # Close the log file
        log_file.close()

        # Hiperparámetros a buscar:
        actor_learning_rate *= 0.8
        
        
        
