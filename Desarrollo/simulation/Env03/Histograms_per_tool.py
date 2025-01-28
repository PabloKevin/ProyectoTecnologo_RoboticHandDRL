import os
import numpy as np
from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv
import matplotlib.pyplot as plt


if __name__ == '__main__':
    tools_of_interest = ["bw_Martillo01.jpg", "empty.png", "bw_Lapicera01.png", "bw_destornillador01.jpg", "bw_tornillo01.jpg"]
    actions=[[]*len(tools_of_interest)]
    for j, tool in enumerate(tools_of_interest):
        env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest=[tool])

        hidden_layers=[32,32] 

        agent = Agent(env=env, hidden_layers=hidden_layers, noise=0.0) 

        agent.load_models()

        episodes = 100

        for i in range(episodes):
            observation = env.reset()
            tool = agent.observer(observation[0]).cpu().detach().numpy() # Takes the image and outputs a tool value
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
            
            actions[j].append(env.state["finger_states"])

    for j, tool in enumerate(tools_of_interest):
        plt.hist(actions[j])
        plt.title(tool)
        plt.show()