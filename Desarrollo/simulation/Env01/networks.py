import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):  
    # Actualiza el Q-value en función del estado y la acción tomada. Controla qué tan bien la ActorNetwork está accionando. 
    # Actualiza el Q-value en función del estado y la acción tomada, actualiza la política.
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name='critic', checkpoint_dir='Desarrollo/simulation/Env01/tmp/td3', learning_rate=10e-3):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions          # cantidad de dimensiones que compone una acción. Ej: brazo robótico con movimiento en 3 ejes (𝑥,𝑦,𝑧), entonces n_actions = 3
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)       # input la DNN: estado + acción, cada uno con sus dimensiones correspondientes
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)                                   # output de la DNN: Q-values según el estado y acción eligidos

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.005)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        print(f"Created Critic Network on device: {self.device}")

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q1 = self.q1(action_value)

        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):      
    # Devuelve la acción a tomar en función del estado
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, learning_rate=10e-3, n_actions=5, n_actions_per_finger=3, name='actor', checkpoint_dir='Desarrollo/simulation/Env01/tmp/td3'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.n_actions_per_finger = n_actions_per_finger
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)       # input de la DNN: estado (con sus dimensiones correspondientes)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions * self.n_actions_per_finger)      # output de la DNN: logits para cada acción

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        print(f"Created Actor Network on device: {self.device}")

        self.to(self.device)

    def forward(self, state):
        # If batch dimension is missing, add it
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.output(x)  # shape: [batch_size, 15]
        
        # Reshape to [batch_size, 5, 3]. 
        # The -1: PyTorch calculates the value for the dimension marked with -1 so that the total number 
        # of elements in the tensor remains constant before and after reshaping.
        logits = logits.view(-1, self.n_actions, self.n_actions_per_finger)

        # Convert to probability distribution along dimension=2 (the 3 discrete actions)
        probs = F.softmax(logits, dim=2)  # shape: [batch_size, 5, 3]

        # Argmax across the 3 possible actions => shape [batch_size, 5]
        actions = T.argmax(probs, dim=2)

        # If batch size = 1, shape is [1,5]; squeeze(0) => [5]
        return actions.squeeze(0).to(T.uint8)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
