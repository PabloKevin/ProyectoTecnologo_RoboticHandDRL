import gym
from gym import spaces
import numpy as np
import os
import cv2
from DataSet_editor import DataSet_editor

class ToolManipulationEnv(gym.Env):
    def __init__(self, image_shape=(256, 256, 1), n_fingers=5):
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
        # Directory containing images
        image_dir = "Desarrollo/simulation/Env01/DataSets/B&W_Tools/"
        
        # List all image files in the directory
        image_files = [f for f in os.listdir(image_dir)]
        # Check how many images are there
        num_images = len(image_files)
        
        # Create a random index
        random_index = np.random.randint(0, num_images)
        
        # Select an image with that index
        selected_image_path = os.path.join(image_dir, image_files[random_index])
        
        # Load the image
        img = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
        # Transform the image so there is a "different" image for each episode
        editor = DataSet_editor()
        img = editor.transform_image(img)
        return img
    
    def _calculate_reward(self, state, action):
        # Define the desired finger state combinations
        combination_1 = [180, 90, 180, 180, 180]  # Thumb closed, index half, others closed
        combination_2 = [180, 90, 90, 180, 180] # Thumb closed, index and middle half, others open
        combination_3 = [90, 90, 90, 90, 90] # All fingers half closed
        combination_4 = [0, 0, 0, 0, 0] # All fingers opened

        # Extract the current finger states
        current_finger_states = state['finger_states']

        # Check for each combination and assign rewards
        if np.array_equal(current_finger_states, combination_1):
            return 2.5
        elif np.array_equal(current_finger_states, combination_2):
            return 2.0
        elif np.array_equal(current_finger_states, combination_3):
            return 1.5
        elif np.array_equal(current_finger_states, combination_4):
            return 1.0 # Check if the reward is enough or should be on other side
        else:
            return -1.0  # Malfunction or undesired combination
    
    def _check_done(self, state):
        # Determine if the task is complete
        # For example, check if the fingers are in the desired position
        # The first approach is just one action and reward, if not the tool would fall maybe. 
        return True

   
import matplotlib.pyplot as plt 
if __name__ == "__main__":
    env = ToolManipulationEnv()
    image = env._get_initial_image()
    
    # Mostrar la imagen usando matplotlib
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Inicial')
    plt.axis('off')  # Ocultar los ejes
    plt.show()

