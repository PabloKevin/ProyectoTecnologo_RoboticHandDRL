from Env03.td3_torch import Agent
from Env03.custom_hand_env import ToolManipulationEnv
import numpy as np
from Env03.networks import ObserverNetwork
import matplotlib.pyplot as plt
import os
import json
import time

class Full_Agent_Pipe():
    def __init__(self, env=None, observer=None, td3_agent=None):
        if env is None:
            self.env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest="all")
        else:
            self.env = env
        if observer is None:
            self.observer = ObserverNetwork()
            self.observer.load_model()
        else:
            self.observer = observer
        if td3_agent is None:
            self.td3_agent = Agent(env=self.env, hidden_layers=[32,32], noise=0.0)
            self.td3_agent.load_models()
        else:
            self.td3_agent = td3_agent
        self.action_file_path = "ACTION_FILE.json"

    def pipe(self, input_img=None, render=True, render_timeout=3):
        if input_img is None:
            input_img = self.env.reset()[0] #image
        
        tool = self.observer(input_img).cpu().detach().numpy() # Takes the image and outputs a tool value

        finger_actions= []
        for f_idx in range(5):
            observation = np.array([tool.item(), f_idx]) 
            finger_actions.append(self.td3_agent.choose_action(observation, validation=True))

        action = {"finger": ["pulgar", "indice", "medio", "anular", "menique"],
                  "position": finger_actions}
        
        if render:
            plt.imshow(input_img.squeeze(), cmap='gray')
            plt.title('Input Image')
            plt.axis('off') 
            if render_timeout is not None:
                plt.show(block=False)
                plt.pause(render_timeout)  # Show plot for x seconds
                plt.close()  # Close the plot window
            else:
                plt.show()

        return action
    
    def write_action(self, action):
        """Writes the predicted action to a file."""
        action_data = {
            "timestamp": time.time(),
            "action": action.tolist()  # Convert NumPy array to list for JSON serialization
        }
        
        with open(self.action_file_path, "w") as file:
            json.dump(action_data, file)
        
        print(f"Action written: {action_data}")
    
    def read_action(self):
        """Reads the last predicted action from file."""
        if not os.path.exists(self.action_file_path):
            return None

        try:
            with open(self.action_file_path, "r") as file:
                action_data = json.load(file)
                return action_data.get("action", None)
        except json.JSONDecodeError:
            return None  # File might be being written to at the moment
        

"""if __name__ == "__main__":
    Full_Agent = Full_Agent_Pipe()

    for _ in range(3):
        action = Full_Agent.pipe()
        print(action)"""


