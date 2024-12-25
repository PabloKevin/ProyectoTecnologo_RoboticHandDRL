import numpy as np
from custom_hand_env import ToolManipulationEnv

def main():
    # Create environment
    env = ToolManipulationEnv()
    
    # Reset environment
    obs = env.reset()
    
    # Simple control loop
    for _ in range(3):
        # Random actions
        action = env.action_space.sample()
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        # Render
        env.render()
        
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main() 