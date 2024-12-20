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

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    # Create an instance of your custom environment
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=5)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128
    warmup = 1000

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.nvec.size,
                  layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, warmup=warmup)

    print("n_actions: ", agent.n_actions)
    writer = SummaryWriter("logs")
    n_games = 10 #10000 recomendados en el video
    
    
    for experiment in range(0,1):
        best_score = 0

        episode_identifier = f"{experiment} - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size} _ RoboticHand"

        #agent.load_models()

        for i in range(n_games):
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

            if i % 10 == 0:
                agent.save_models()

            print(f"Episode: {i} Score: {score}")

        # Hiperparámetros a buscar:
        actor_learning_rate *= 0.8
        
