from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv
import numpy as np
from networks import ObserverNetwork, ActorNetwork
import matplotlib.pyplot as plt
from SAM_pipe import Segmentator
import os
import cv2
import torch as T
import torch.nn.functional as F

class Full_Agent_Pipe():
    def __init__(self, env=None, segmentator=None, observer=None, actor=None, checkpoint_dir="Desarrollo/simulation/Env04/models_params_weights/"): 
        """if env is None:
            self.env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=1, images_of_interest="all")
        else:
            self.env = env"""
        if segmentator is None:
            self.segmentator = Segmentator(checkpoint_dir=checkpoint_dir+"SAM/sam_vit_b_01ec64.pth") # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto.
        else:
            self.segmentator = segmentator
        if observer is None:
            self.observer = ObserverNetwork(checkpoint_dir=checkpoint_dir+"observer", name="observer_final_v7") # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
            self.observer.load_model()
            self.observer.eval() #importante para apagar las capas de dropout.
        else:
            self.observer = observer
        if actor is None:
            self.actor = ActorNetwork(checkpoint_dir=checkpoint_dir+"td3", hidden_layers=[64,32,16]) # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
            #self.actor.checkpoint_file = checkpoint_dir + "td3/actor_episode_25000"
            self.actor.checkpoint_file =  "model_weights_docs/td3/v2_trainset/actor_episode_6500" # Para visualizar aprendizaje con pesos en episodios bajos
            self.actor.load_checkpoint()
            self.actor.eval() # no estoy seguro si es necesario, pero es estándar.
        else:
            self.actor = actor

    def pipe(self, input_img=None, dataset_name="RawTools", render=True, render_timeout=3):
        if input_img is None:
            image_dir = "Desarrollo/simulation/Env04/DataSets/" + dataset_name
            image_files = os.listdir(image_dir)
            image_file = np.random.choice(image_files)
            img_path = os.path.join(image_dir, image_file)
            input_img = cv2.imread(img_path)

        bw_mask = self.segmentator.predict(input_img, render=False)
        bw_mask = np.expand_dims(bw_mask, axis=0)
        print("bw_mask", bw_mask.shape)
        logits = self.observer(bw_mask) # Takes the image and outputs a tool value
        probs  = F.softmax(logits, dim=-1).cpu().detach().numpy() 
        tool = np.argmax(probs) # tool value
        print("tool", tool)

        finger_actions= []
        for f_idx in range(5):
            observation = T.tensor(np.append(logits.cpu().detach().numpy(), float(f_idx)), dtype=T.float).to(self.actor.device)
            pred_action = self.actor(observation).item()
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

        return action, bw_mask
   
if __name__ == "__main__":
    Full_Agent = Full_Agent_Pipe()

    for _ in range(1):
        action = Full_Agent.pipe(dataset_name="RawTools_test")
        print(action)


