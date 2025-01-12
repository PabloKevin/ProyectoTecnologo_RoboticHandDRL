import gym
from gym import spaces
import numpy as np
import os
import cv2
from DataSet_editor import DataSet_editor
import matplotlib.pyplot as plt 
import torch

class ToolManipulationEnv(gym.Env):
    def __init__(self, image_shape=(256, 256, 1), n_fingers=5, n_choices_per_finger=3):
        super(ToolManipulationEnv, self).__init__()
        
        # Observation space: image + finger states
        self.image_shape = image_shape
        self.n_fingers = n_fingers
        self.n_choices_per_finger = n_choices_per_finger
        self.observation_space = spaces.Dict({
            # Image in black and white, so 0 = Black, 1 = White 
            'image': spaces.MultiDiscrete([2] * np.prod(self.image_shape)),
            
            'finger_states': spaces.MultiDiscrete([self.n_choices_per_finger] * self.n_fingers)
        })
        self.combinations_of_interest = [[2, 1, 2, 2, 2], # Thumb closed, index half, others closed,
                                         [2, 1, 1, 2, 2], # Thumb closed, index and middle half, others open
                                         [1, 1, 1, 1, 1], # All fingers half closed
                                         [0, 0, 0, 0, 0]  # All fingers opened
        ]
        
        self.reward_weights = { "reward_alpha" : 1,
                               "individual_finger_reward" : [0.25, 0.25, 0.25, 0.25, 0.25],
                               "reward_beta" : [5.9, 5.3, 2, 5.15, 3.0, 0.0],
                               "reward_gamma" : [1.0, 0.70, -1.2, 0.6],
                               "repeated_action_penalty" : [0.8, 0.2]
        }
        
        # Action space: 3 actions per finger. If it's going to be a continuous action space, it should be a Box space
        self.action_space = spaces.MultiDiscrete([self.n_choices_per_finger] * self.n_fingers)
        
        # Initialize state
        self.state = {
            'image': np.zeros(self.image_shape, dtype=np.uint8),
            'finger_states': np.zeros(self.n_fingers, dtype=np.uint8)
        }

        self.reward = 0
        self.wrong_action_cntr = 0
        self.previous_action = np.zeros(self.n_fingers, dtype=np.uint8)
    
    def get_observation_space_shape(self):
        # probar con imagen ordenada
        #return (self.image_shape[0], self.image_shape[1], self.image_shape[2])
        return self.image_shape
        #return (self.image_shape[0], self.image_shape[1])

    def reset(self):
        # Reset the environment to an initial state
        self.state['image'] = self._get_initial_image()
        self.state['finger_states'] = np.zeros(self.n_fingers, dtype=np.float16)
        self.reward = 0
        observation = self.state['image']

        return observation
    
    def step(self, action):
        """
        # Update finger states based on action
        for i in range(self.n_fingers):
            if action[i] == 0:
                self.state['finger_states'][i] = 0  # Open
            elif action[i] == 1:
                self.state['finger_states'][i] = 90  # Medium closed
            elif action[i] == 2:
                self.state['finger_states'][i] = 180  # Fully closed
        """
        # Update the finger states
        self.state['finger_states'] = action

        # Calculate reward
        self.reward = self._calculate_reward(self.state, action)
        
        # Flatten the image and concatenate with finger states
        #flattened_image = self.state['image'].flatten()
        #observation = np.concatenate((flattened_image, action))
        # probar unicamente con la imagen
        observation = self.state['image']
        
        return observation, self.reward, {}
    
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
        image_dir = "Desarrollo/simulation/Env02/DataSets/B&W_Tools/"
        
        # List all image files in the directory
        image_files = [f for f in os.listdir(image_dir)]
        # Check how many images are there
        num_images = len(image_files)
        
        # Create a random index
        random_index = np.random.randint(0, num_images)
        
        # Select an image with that index
        #selected_image_path = os.path.join(image_dir, image_files[random_index])
        selected_image_path = "Desarrollo/simulation/Env02/DataSets/B&W_Tools/bw_Martillo01.jpg"
        
        # Load the image
        img = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
        # Transform the image so there is a "different" image for each episode
        editor = DataSet_editor()
        img = editor.transform_image(img)
        #white_pixels = np.argwhere(img == 255)
        #print(len(white_pixels))
        # Convert 255 pixels to 1
        img[img < 255/2] = 0  
        img[img >=  255/2] = 1
        #file = "Desarrollo/simulation/Env01/img.txt"
        #np.savetxt(file, img, fmt="%d", delimiter=" ") 
        img = np.expand_dims(img, axis=-1)
        return img
    
    def _calculate_reward(self, state, probs):
        reward = 0
        action = self.probs2actions(probs)
        # Extract the current finger states
        #current_finger_states = state['finger_states']
        # Find a way to "subtract" reward if the object falls, or if the object is too large
        # to use few fingers, so it doesn't bias towards using the first combination. 
        # Watch the quantity of white pixels.
        n_white_pixels = len(np.argwhere(self.state['image'] == 1))
        negative_reward = np.sqrt(n_white_pixels/1000)
        # Check for each combination and assign rewards
        # usar diccionarios para pesos y grados, como buena práctica. 
        if np.array_equal(action, self.combinations_of_interest[0]):
            if n_white_pixels == 0:
                reward += (self.reward_weights["reward_beta"][4] - negative_reward * self.reward_weights["reward_gamma"][3]) * self.reward_weights["reward_alpha"]
            else:
                reward += (self.reward_weights["reward_beta"][0] - negative_reward * self.reward_weights["reward_gamma"][0]) * self.reward_weights["reward_alpha"]
        elif np.array_equal(action, self.combinations_of_interest[1]):
            if n_white_pixels == 0:
                reward += (self.reward_weights["reward_beta"][4] - negative_reward * self.reward_weights["reward_gamma"][3]) * self.reward_weights["reward_alpha"]
            else:
                reward += (self.reward_weights["reward_beta"][1] - negative_reward * self.reward_weights["reward_gamma"][1]) * self.reward_weights["reward_alpha"]
        elif np.array_equal(action, self.combinations_of_interest[2]):
            if n_white_pixels == 0:
                reward += (self.reward_weights["reward_beta"][4] - negative_reward * self.reward_weights["reward_gamma"][3]) * self.reward_weights["reward_alpha"]
            else:
                reward += (self.reward_weights["reward_beta"][2] - negative_reward * self.reward_weights["reward_gamma"][2]) * self.reward_weights["reward_alpha"]
        elif np.array_equal(action, self.combinations_of_interest[3]):
            #return 1.0 - negative_reward * 1
            if n_white_pixels == 0:
                reward += self.reward_weights["reward_beta"][3] * self.reward_weights["reward_alpha"]
            else:
                # if it continues to be biased, try to use a more negative reward here
                reward += (self.reward_weights["reward_beta"][4] - negative_reward * self.reward_weights["reward_gamma"][3]) * self.reward_weights["reward_alpha"]
        # Check if the reward is enough or should be on other side
        else:
            reward += self.reward_weights["reward_beta"][5] * self.reward_weights["reward_alpha"]  # Malfunction or undesired combination
        
            # reward for each finger so as to motivate the combinations
            if action[0] == 2:
                reward += self.reward_weights["individual_finger_reward"][0]
            if action[1] == 1:
                reward += self.reward_weights["individual_finger_reward"][1]
            if action[2] == 1:
                reward += self.reward_weights["individual_finger_reward"][2]
            if action[3] == 2:
                reward += self.reward_weights["individual_finger_reward"][3]
            if action[4] == 2:
                reward += self.reward_weights["individual_finger_reward"][4]
        
        # Penalize wrong actions
        if reward < self.reward_weights["repeated_action_penalty"][0]:
            self.wrong_action_cntr += 1
            reward -= self.wrong_action_cntr * self.reward_weights["repeated_action_penalty"][1]
            if np.array_equal(action, self.previous_action):
                reward -= self.reward_weights["repeated_action_penalty"][1]
        else:
            self.wrong_action_cntr = 0

        self.previous_action = action  # Actualizar la acción anterior
        # Final reward
        return reward

    
    def probs2actions(self, probs):
        print(probs)
        action = np.argmax(probs, axis=1)
        print(action)
        self.state['finger_states'] = action
        return action
   
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
#"""