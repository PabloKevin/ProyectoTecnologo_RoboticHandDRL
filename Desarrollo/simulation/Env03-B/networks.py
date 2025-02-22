import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

# Critic Network
class CriticNetwork(nn.Module):
    # Actualiza el Q-value en función del estado y la acción tomada. Controla qué tan bien la ActorNetwork está accionando. 
    # Actualiza el Q-value en función del estado y la acción tomada, actualiza la política.
    # Dado que es un single-step episode, predice solamente el Q-value inmediato para la acción tomada.
    def __init__(self, input_dims,  conv_channels=[16, 32, 64], hidden_layers=[128,64], name='critic', checkpoint_dir='Desarrollo/simulation/Env03/tmp/td3', 
                 learning_rate=0.001):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims # check if this is necessary
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')


        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(conv_channels[2] * (input_dims[0] // 8) * (input_dims[1] // 8), hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], 1)
        self.fc3 = nn.Linear(1 + 2, 1)


        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Critic Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, state, action):
        img, f_idx = state
        x = F.leaky_relu(self.conv1(img))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.reshape((x.size(0), -1))  # Flatten each sample in the batch
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.cat([x, f_idx, action], dim=1)
        q_value = self.fc3(x)

        return q_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


# Actor Network
class ActorNetwork(nn.Module):
    # Devuelve la acción a tomar en función del estado
    def __init__(self, input_dims, n_actions, hidden_layers=[64,32], name='actor', checkpoint_dir='Desarrollo/simulation/Env03/tmp/td3', learning_rate=0.001):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims # check if this is necessary
        self.checkpoint_dir = checkpoint_dir
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        
        self.fc1 = nn.Linear(input_dims, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], n_actions)
        
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Actor Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        #print(f"Shape after conv3: {x.shape}")
        # Check if the input is a batch or a single image
        action = torch.tanh(self.fc3(x)) # output between (-1,1)
            
        return action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


# Observer Network
class ObserverNetwork(nn.Module):
    # Devuelve la acción a tomar en función del estado
    def __init__(self, input_dims = (256, 256, 1), output_dims = 1, conv_channels=[16, 32, 64], hidden_layers=[64,8], name='observer', checkpoint_dir='Desarrollo/simulation/Env03/tmp/observer', learning_rate=0.001):
        super(ObserverNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_supervised')

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(conv_channels[2] * (input_dims[0] // 8) * (input_dims[1] // 8), hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_dims)
        
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Observer Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, img):
        img = torch.tensor(img, dtype=torch.float).to(self.device)
        x = F.leaky_relu(self.conv1(img))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        #print(f"Shape after conv3: {x.shape}")
        # Check if the input is a batch or a single image
        if len(x.shape) == 4:  # Batch case: [batch_size, channels, height, width]
            x = x.reshape((x.size(0), -1))  # Flatten each sample in the batch
        elif len(x.shape) == 3:  # Single image case: [channels, height, width]
            x = x.reshape(-1)  # Flatten the single image
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        tool_reg = F.leaky_relu(self.fc3(x))
            
        return tool_reg # Tool regresion

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        print("Successfully loaded observer model")
