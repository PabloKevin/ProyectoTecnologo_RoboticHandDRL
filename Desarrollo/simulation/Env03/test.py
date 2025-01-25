import time
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv


if __name__ == '__main__':

    if not os.path.exists("Desarrollo/simulation/Env03/tmp/td3"):
        os.makedirs("Desarrollo/simulation/Env03/tmp/td3")

    # Create an instance of your custom environment
    env = ToolManipulationEnv(image_shape=(256, 256, 1))
# Ir probando con numeros más simples para que lleve menos tiempo. Dado que el problemas es más simple,
# usar menos neuronas, probablemente no necesite tantas imágenes para aprender. Quizá probar con 1 sola capa.

    actor_learning_rate = 0.003 #0.001    1.0
    critic_learning_rate = 0.0008 #0.001   0.001
    batch_size = 64 #128

    hidden_layers=[32,16] #256
    warmup = 0
    env.reward_weights["reward_alpha"] = 1

    # Reduce the replay buffer size
    max_size = 10000  # Adjust this value based on your memory capacity

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.05, #tau=0.005
                  env=env, n_actions=env.n_fingers, hidden_layers=hidden_layers, 
                  batch_size=batch_size, warmup=warmup, max_size=max_size) 

    agent.load_models()

    episodes = 1000 #10000
    scores = []
    right_comb = 0
    for i in range(episodes):
        observation = env.reset()
        tool = agent.observer(observation[0]).cpu().detach().numpy() # Takes the image and outputs a tool value
        observation = np.array([tool.item(), observation[1]]) # [tool, f_idx]
        done = False
        score = 0
        right_comb_ctr = 0
        while not done:
            action = agent.choose_action(observation) 
            next_observation, reward, done, info = env.step(action)
            next_observation = np.array([tool.item(), next_observation[1]]) # [tool, f_idx]
            score += reward
            observation = next_observation
            if reward > -2/3:
                right_comb_ctr += 1
        
        #env.render(timeout=None)
        scores.append(score)
        if right_comb_ctr == 5:
            right_comb += 1

        # Log the information to the text file
        print(f"Episode: {i}; Score: {score}; Tool: {tool}; Action: {env.complete_action()}")

    print(f"Test episodes: {episodes}")
    print(f"Average score: {np.mean(scores)}")
    print(f"Accuracy: {right_comb/len(scores)}")