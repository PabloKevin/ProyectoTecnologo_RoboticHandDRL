import gym
from gym import spaces
import numpy as np
import os
import cv2
from DataSet_editor import DataSet_editor
import matplotlib.pyplot as plt 

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
            'finger_states': spaces.Box(low=0, high=180, shape=(self.n_fingers,), dtype=np.uint8)
        })
        
        # Action space: 3 actions per finger. If it's going to be a continuous action space, it should be a Box space
        self.action_space = spaces.MultiDiscrete([3] * self.n_fingers)
        
        # Initialize state
        self.state = {
            'image': np.zeros(self.image_shape, dtype=np.uint8),
            'finger_states': np.zeros(self.n_fingers, dtype=np.float16) #llevar a uint8
        }

        self.reward = 0
    
    def get_observation_space_shape(self):
        # first approach
        # return (self.image_shape[0], self.image_shape[1], self.image_shape[2] + self.n_fingers)
        # probar usar un flatten, usar todo en un vector
        return (np.prod(self.image_shape) + self.n_fingers,)

    def reset(self):
        # Reset the environment to an initial state
        self.state['image'] = self._get_initial_image()
        self.state['finger_states'] = np.zeros(self.n_fingers, dtype=np.float16)
        self.reward = 0
        
        # Flatten the image and concatenate with finger states
        flattened_image = self.state['image'].flatten()
        observation = np.concatenate((flattened_image, self.state['finger_states']))
        
        return observation
    
    def step(self, action):
        # Update finger states based on action
        for i in range(self.n_fingers):
            if action[i] == 0:
                self.state['finger_states'][i] = 0  # Open
            elif action[i] == 1:
                self.state['finger_states'][i] = 90  # Medium closed
            elif action[i] == 2:
                self.state['finger_states'][i] = 180  # Fully closed
        
        # Calculate reward
        self.reward = self._calculate_reward(self.state, action)
        
        # Check if the task is done
        done = self._check_done(self.state)
        
        # Flatten the image and concatenate with finger states
        flattened_image = self.state['image'].flatten()
        observation = np.concatenate((flattened_image, self.state['finger_states']))
        
        return observation, self.reward, done, {}
    
    def render(self, timeout=None):
        plt.imshow(self.state['image'], cmap='gray')
        plt.title('Episode Image')
        plt.axis('off')  # Ocultar los ejes
        
        # Add text annotation for finger states
        finger_states_text = f"Finger states after action: {self.state['finger_states']}"
        plt.text(-12, 266, finger_states_text, color='white', fontsize=12, 
                 bbox=dict(facecolor='black', alpha=0.7))
        plt.text(-12, 286, f"Reward: {self.reward}", color='white', fontsize=12, 
                 bbox=dict(facecolor='black', alpha=0.7))
        
        if timeout is not None:
            plt.show(block=False)
            plt.pause(timeout)  # Show plot for 0.5 seconds
            plt.close()  # Close the plot window
        else:
            plt.show()
    
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
        #white_pixels = np.argwhere(img == 255)
        #print(len(white_pixels))
        return img
    
    def _calculate_reward(self, state, action):
        # Define the desired finger state combinations
        combination_1 = [180, 90, 180, 180, 180]  # Thumb closed, index half, others closed
        combination_2 = [180, 90, 90, 180, 180] # Thumb closed, index and middle half, others open
        combination_3 = [90, 90, 90, 90, 90] # All fingers half closed
        combination_4 = [0, 0, 0, 0, 0] # All fingers opened

        # Extract the current finger states
        current_finger_states = state['finger_states']
        # Find a way to "subtract" reward if the object falls, or if the object is too large
        # to use few fingers, so it doesn't bias towards using the first combination. 
        # Watch the quantity of white pixels.
        n_white_pixels = len(np.argwhere(self.state['image'] == 255))
        negative_reward = np.sqrt(n_white_pixels/1000)
        # Check for each combination and assign rewards
        # usar diccionarios para pesos y grados, como buena práctica.
        if np.array_equal(current_finger_states, combination_1):
            return 3.5 - negative_reward * 1
        elif np.array_equal(current_finger_states, combination_2):
            return 2.5 - negative_reward * 0.5
        elif np.array_equal(current_finger_states, combination_3):
            return 1.5 - negative_reward * 0.25
        elif np.array_equal(current_finger_states, combination_4):
            #return 1.0 - negative_reward * 1
            if n_white_pixels == 0:
                return 2.75
            else:
                # if it continues to be biased, try to use a more negative reward here
                return 1.0 - negative_reward * 0.5
        # Check if the reward is enough or should be on other side
        else:
            return -1.0  # Malfunction or undesired combination
    
    def _check_done(self, state):
        # Determine if the task is complete
        # For example, check if the fingers are in the desired position
        # The first approach is just one action and reward, if not the tool would fall maybe. 
        return True

   
"""
if __name__ == "__main__":
    env = ToolManipulationEnv()
    env.reset()
    action = env.action_space.sample()  # Randomly sample a valid action
    #print(env.step(action))
    print(env.step(action)[0].shape)
    print(env.reset().shape)
    print(env.get_observation_space_shape())
    env.render()

"""