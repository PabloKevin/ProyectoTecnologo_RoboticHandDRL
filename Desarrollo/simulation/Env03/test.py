import time
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv
from networks import ObserverNetwork


if __name__ == '__main__':

    if not os.path.exists("Desarrollo/simulation/Env03/tmp/td3"):
        os.makedirs("Desarrollo/simulation/Env03/tmp/td3")

    #images_of_interest = ["bw_Martillo01.jpg", "empty.png", "bw_Lapicera01.png", "bw_destornillador01.jpg", "bw_tornillo01.jpg"]
    images_of_interest = "all"
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest=images_of_interest)
    observer = ObserverNetwork() # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
    observer.load_model()

    hidden_layers=[32,32] 

    agent = Agent(env=env, hidden_layers=hidden_layers, noise=0.0) 

    agent.load_models()

    episodes = 1
    scores = []
    right_comb = 0
    for i in range(episodes):
        observation = env.reset()
        tool = observer(observation[0]).cpu().detach().numpy() # Takes the image and outputs a tool value
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
            if reward > -2/3:
                right_comb_ctr += 1
        
        env.render(timeout=None)
        scores.append(score)
        if right_comb_ctr == 5:
            right_comb += 1

        # Log the information to the text file
        #print(f"Episode: {i}; Score: {score}; Tool: {tool}; Action: {env.complete_action()}")

    print(f"Test episodes: {episodes}")
    print(f"Average score: {np.mean(scores)}")
    print(f"Accuracy: {right_comb/len(scores)}")