import gym
#from gym import spaces
import numpy as np
import os
import cv2
from DataSet_editor import DataSet_editor
import matplotlib.pyplot as plt 
from observerRGB import ObserverNetwork, MyImageDataset
import torch
import torch.nn.functional as F

class ToolManipulationEnv(gym.Env):
    def __init__(self, image_shape=(256, 256, 1), n_fingers=1, images_of_interest="all", dataset_dir="Desarrollo/simulation/Env05/DataSets/", dataset_name="TrainSet_masks"):
        super(ToolManipulationEnv, self).__init__()
        
        self.image_shape = image_shape
        self.n_fingers = n_fingers
        self.images_of_interest = images_of_interest
        self.datasets = []
        for dataset in dataset_name:
            self.datasets.append(MyImageDataset(dataset_dir+dataset, name=dataset))
        self.observer = ObserverNetwork(checkpoint_dir='Desarrollo/simulation/Env05/tmp/observer_backup', name="observer_best_test_logits_best2", #"observer_best_test_medium02"
                                        conv_channels = [16, 32, 64], hidden_layers = [64, 32, 16])
        self.observer.checkpoint_file = os.path.join(self.observer.checkpoint_dir, self.observer.name)
        self.observer.load_model()
        self.observer.eval()

        self.combinations_of_interest = [[2, 1, 2, 2, 2], # Thumb closed, index half, others closed,
                                         [2, 0.67, 1, 2, 2], # Thumb closed, index and middle half, others open
                                         [1, 1, 1, 1, 1], # All fingers half closed
                                         [0, 0, 0, 0, 0]  # All fingers opened
        ]
        
        self.reward_weights = { "reward_alpha" : 1,
                               "individual_finger_reward" : [0.25, 0.25, 0.25, 0.25, 0.25],
                               "right" : 3.0,
                               "wrong" : 1.0,
                               "awful" : 0.0,
                               "repeated_action_penalty" : [1.2, 0.2, -3.0]
        }
        
        self.state = {
            'image': np.zeros(self.image_shape, dtype=np.uint8),
            'label': 0.0,
            'tool': 0.0,
            'f_idx': 0.0,
            'finger_state': 0.0,
            'finger_states': np.zeros(5, dtype=np.uint8),
            'best_combination': np.zeros(5, dtype=np.uint8)
        }
        self.done = False
        self.reward = 0
        self.wrong_action_cntr = 0
        self.previous_action = np.zeros(self.n_fingers, dtype=np.uint8)
        

    def reset(self):
        # Reset the environment to an initial state
        self.state['image'], self.state['label'] = self._get_initial_image()
        self.state['tool'] = self._get_tool(self.state['image'])
        self.state['f_idx'] = 0.0
        self.state['finger_state'] = 0.0
        self.state['finger_states'] = np.zeros(5, dtype=np.float64)
        self.state['best_combination'] = self._calculate_best_combination(self.state['label'])
        self.done = False
        self.reward = 0
        observation = np.concatenate([self.state['tool'], np.array([self.state['f_idx']])])
        return observation
    
    def step(self, action):
        # Update the finger states
        self.state['finger_state'] = action
        self.state['finger_states'][int(self.state['f_idx'])] = action

        # Calculate reward
        self.reward = self._calculate_reward(self.state, action)
        
        self.state['f_idx'] += 1 
        if self.state['f_idx'] == 5:
            self.done = True
        else:
            self.done = False

        next_observation = np.concatenate([self.state['tool'], np.array([self.state['f_idx']])])
        info = {} # no le he encontrado utilidad, pero podría ser util.
        
        return next_observation, self.reward, self.done, info
    
    def render(self, timeout=None):
        plt.imshow(self.state['image'].squeeze(), cmap='gray')
        plt.title('Episode Image')
        plt.axis('off')  # Ocultar los ejes
        
        # Add text annotation for finger states
        finger_states_text = f"Finger states after action: {self.complete_action()[1]}"
        plt.text(-12, 266, finger_states_text, color='white', fontsize=12, 
                 bbox=dict(facecolor='black', alpha=0.7))
        plt.text(-12, 286, f"Score: {self.reward}", color='white', fontsize=12, 
                 bbox=dict(facecolor='black', alpha=0.7))
        
        if timeout is not None:
            plt.show(block=False)
            plt.pause(timeout)  # Show plot for x seconds
            plt.close()  # Close the plot window
        else:
            plt.show()
    
    def _get_initial_image(self):
        ds = np.random.randint(0, len(self.datasets))
        idx = np.random.randint(0, len(self.datasets[ds].image_files))
        
        if self.images_of_interest == "all":
            return self.datasets[ds].__getitem__(idx)
        else:
            img, label = self.datasets[ds].__getitem__(idx)
            key = next((k for k,v in self.datasets[ds].label_mapping.items() if v == label), None)
            while key not in self.images_of_interest:
                idx = np.random.randint(0, len(self.datasets[ds].image_files))
                img, label = self.datasets[ds].__getitem__(idx)
                key = next((k for k,v in self.datasets[ds].label_mapping.items() if v == label), None)
            #print("img_label:", label)
            return img, label
    
    def _get_tool(self, img):
        img = img.to(self.observer.device)
        with torch.no_grad():
            logits = self.observer(img)           # (B,10)
            #probs  = F.softmax(logits, dim=-1)        # (B,10) if you need actual probs
        return logits.cpu().detach().numpy() # Takes the image and outputs a tool value
    
    def _calculate_best_combination(self, label):
        if label == 0.0:
            best_combination = self.combinations_of_interest[3]
        elif label < 3.5:
            best_combination = self.combinations_of_interest[0]
        elif label < 6.5:
            best_combination = self.combinations_of_interest[1]
        else:
            best_combination = self.combinations_of_interest[2]
        return best_combination

    def _calculate_reward(self, state, action):
        action = action + 1 # Porque la salida está en (-1,1) y así pasamos a (0,2)
        self.reward = - np.power(abs(action - state['best_combination'][int(state['f_idx'])])+1,2)
        
        if self.reward > 2/3:
            self.wrong_action_cntr += 1
            self.reward += self.wrong_action_cntr * self.reward * 0.25
        else:
            self.wrong_action_cntr = 0
        
        
        
        """# reward for each finger so as to motivate the combinations
        if action[0] == 2:
            reward += self.reward_weights["individual_finger_reward"][0] * self.reward_weights["reward_alpha"]
        if action[1] == 1:
            reward += self.reward_weights["individual_finger_reward"][1] * self.reward_weights["reward_alpha"]
        if action[2] == 1:
            reward += self.reward_weights["individual_finger_reward"][2] * self.reward_weights["reward_alpha"]
        if action[3] == 2:
            reward += self.reward_weights["individual_finger_reward"][3] * self.reward_weights["reward_alpha"]
        if action[4] == 2:
            reward += self.reward_weights["individual_finger_reward"][4] * self.reward_weights["reward_alpha"]"""
        
        """# Penalize wrong actions
        if reward < self.reward_weights["repeated_action_penalty"][0] and reward > self.reward_weights["repeated_action_penalty"][2]:
            self.wrong_action_cntr += 1
            reward -= self.wrong_action_cntr * self.reward_weights["repeated_action_penalty"][1]
            if np.array_equal(action, self.previous_action):
                reward -= self.reward_weights["repeated_action_penalty"][1]
        else:
            self.wrong_action_cntr = 0

        self.previous_action = action  # Actualizar la acción anterior"""
        # Final reward
        return self.reward
    
    def complete_action(self):
        c_action = []
        o_action = []
        for action in self.state['finger_states']:
            if action < 2/3 - 1:
                c_action.append(0)
            elif action  < 2*2/3 - 1:
                c_action.append(1)
            else:
                c_action.append(2)
        for action in self.state['finger_states']:
            o_action.append(round(action + 1, 2))
        return o_action, c_action

   
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