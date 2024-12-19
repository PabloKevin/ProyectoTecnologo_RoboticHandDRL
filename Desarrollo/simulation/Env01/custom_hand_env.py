import gym
from gym import spaces
import numpy as np

class ToolManipulationEnv(gym.Env):
    def __init__(self, image_shape=(64, 64, 1), n_fingers=5):
        super(ToolManipulationEnv, self).__init__()
        
        # Observation space: image + finger states
        self.image_shape = image_shape
        self.n_fingers = n_fingers
        self.observation_space = spaces.Dict({
            # Image in black and white, so 0 = Black, 1 = White , uint8 enough
            'image': spaces.Box(low=0, high=1, shape=self.image_shape, dtype=np.uint8),
            # Finger states in degrees, so 0 = 0 degrees, 180 = 180 degrees, flaot16 enough and scalable if wanted float numbers
            'finger_states': spaces.Box(low=0, high=180, shape=(self.n_fingers,), dtype=np.float16)
        })
        
        # Action space: 3 actions per finger. If it's going to be a continuous action space, it should be a Box space
        self.action_space = spaces.MultiDiscrete([3] * self.n_fingers)
        
        # Initialize state
        self.state = {
            'image': np.zeros(self.image_shape, dtype=np.uint8),
            'finger_states': np.zeros(self.n_fingers, dtype=np.float16)
        }
        
    def reset(self):
        # Reset the environment to an initial state
        self.state['image'] = self._get_initial_image()
        self.state['finger_states'] = np.zeros(self.n_fingers, dtype=np.float16)
        return self.state
    
    def step(self, action):
        # Update finger states based on action
        for i in range(self.n_fingers):
            if action[i] == 0:
                self.state['finger_states'][i] = 0.0  # Open
            elif action[i] == 1:
                self.state['finger_states'][i] = 90.0  # Medium closed
            elif action[i] == 2:
                self.state['finger_states'][i] = 180.0  # Fully closed
        
        # Calculate reward
        reward = self._calculate_reward(self.state, action)
        
        # Check if the task is done
        done = self._check_done(self.state)
        
        return self.state, reward, done, {} 
        # the {} is for the info dictionary, it's empty because we don't need to pass any info, usefull in debugging
    
    def render(self, mode='human'):
        # Optionally implement rendering
        pass
    
    def _get_initial_image(self):
        # Generate or load the initial image
        return np.random.randint(0, 256, self.image_shape, dtype=np.uint8)
    
    def _calculate_reward(self, state, action):
        # Implement a reward function based on the state and action
        # For example, reward could be based on how well the fingers are positioned to use the tool
        return 0.0
    
    def _check_done(self, state):
        # Determine if the task is complete
        # For example, check if the fingers are in the desired position
        # The first approach is just one action and reward, if not the tool would fall maybe. 
        return True