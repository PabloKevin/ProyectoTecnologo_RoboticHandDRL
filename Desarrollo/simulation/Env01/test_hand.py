import numpy as np
from custom_hand_env import RoboticHandEnv

def main():
    # Create environment
    env = RoboticHandEnv()
    
    # Reset environment
    obs = env.reset()
    
    # Simple control loop
    for _ in range(1000):
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