import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
    layer1_size = 64 #32/64
    layer2_size = 32
    warmup = 900

    # Reduce the replay buffer size
    max_size = 10000  # Adjust this value based on your memory capacity

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.get_observation_space_shape(), env=env, n_actions=env.n_fingers,
                  layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, warmup=warmup,
                  max_size=max_size)  # Pass the max_size to the Agent

    n_games = 6 
    best_score = 0
    
    agent.load_models()

    for i in range(n_games):
        #print("n_actions", agent.n_actions) #solo por curiosidad, porque en el código usaba =2 por defecto, pero en este entorno es 8.
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, validation=True) #Importante el validation=True para que no entre en warmup y haga acciones randoms
            next_observation, reward, done, info = env.step(action)
            #env.render(timeout=2)
            score += reward
            observation = next_observation
            time.sleep(0.01)

        #print(f"Episode: {i}; Score: {score}; Action: {env.state['finger_states']}")
        print(f"Episode: {i}; Score: {score}; Action: {action}")