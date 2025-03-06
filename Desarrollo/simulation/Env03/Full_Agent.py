from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv
import numpy as np
from networks import ObserverNetwork
import matplotlib.pyplot as plt
from SAM import Segmentator

class Full_Agent_Pipe():
    def __init__(self, env=None, segmentator=None, observer=None, td3_agent=None, checkpoint_dir="tmp/"):
        if env is None:
            self.env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest="all")
        else:
            self.env = env
        if segmentator is None:
            self.segmentator = None
        else:
            self.segmentator = segmentator
        if observer is None:
            self.observer = ObserverNetwork(checkpoint_dir=checkpoint_dir+"observer") # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
            self.observer.load_model()
        else:
            self.observer = observer
        if td3_agent is None:
            self.td3_agent = Agent(env=self.env, hidden_layers=[32,32], noise=0.0, checkpoint_dir=checkpoint_dir+"td3") # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
            self.td3_agent.load_models()
        else:
            self.td3_agent = td3_agent

    def pipe(self, input_img=None, render=True, render_timeout=3):
        if input_img is None:
            input_img = self.env.reset()[0] #image
        
        tool = self.observer(input_img).cpu().detach().numpy() # Takes the image and outputs a tool value
        print("tool", tool)

        finger_actions= []
        for f_idx in range(5):
            observation = np.array([tool.item(), float(f_idx)])
            pred_action = self.td3_agent.choose_action(observation, validation=True).item()
            finger_actions.append(pred_action + 1) # pred_action (-1, 1) +1 -> (0,2) intervals and action spaces

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
    
"""if __name__ == "__main__":
    Full_Agent = Full_Agent_Pipe()

    for _ in range(3):
        action = Full_Agent.pipe()
        print(action)"""


