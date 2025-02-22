import numpy as np
from custom_hand_env import ToolManipulationEnv
from matplotlib import pyplot as plt

def main():
    # Create environment
    env = ToolManipulationEnv()
    
    # Reset environment
    obs = env.reset()
    actions = []
    
    
    
    # Simple control loop
    for _ in range(100):
        # Random actions
        action = env.action_space.sample()
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        # Render
        #env.render()
        
        if done:
            obs = env.reset()

        actions.append(action[0])


    plt.hist(actions)
    plt.show()
    print(actions)

if __name__ == "__main__":
    main() 