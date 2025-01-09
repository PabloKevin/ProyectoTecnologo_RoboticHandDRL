import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# Critic Network
class CriticNetwork(nn.Module):
    # Actualiza el Q-value en función del estado y la acción tomada. Controla qué tan bien la ActorNetwork está accionando. 
    # Actualiza el Q-value en función del estado y la acción tomada, actualiza la política.
    # Dado que es un single-step episode, predice solamente el Q-value inmediato para la acción tomada.
    def __init__(self, input_dims, n_actions, conv_channels=[16, 32, 64], hidden_size=256, name='critic', checkpoint_dir='Desarrollo/simulation/Env02/tmp/td3', learning_rate=0.001):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims # check if this is necessary
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(conv_channels[2] * (input_dims[0] // 8) * (input_dims[1] // 8) + n_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Critic Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1)
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


# Actor Network
class ActorNetwork(nn.Module):
    # Devuelve la acción a tomar en función del estado
    def __init__(self, input_dims, n_actions, n_choices_per_finger, conv_channels=[16, 32, 64], hidden_size=256, name='actor', checkpoint_dir='Desarrollo/simulation/Env02/tmp/td3', learning_rate=0.001):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims # check if this is necessary
        self.checkpoint_dir = checkpoint_dir
        self.n_actions = n_actions
        self.n_choices_per_finger = n_choices_per_finger
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(conv_channels[2] * (input_dims[0] // 8) * (input_dims[1] // 8), hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions * n_choices_per_finger)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Actor Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, state):
        state = state.to(self.device)
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x).view(self.n_actions, self.n_choices_per_finger)
        actions = torch.argmax(logits, dim=1)  # Discrete action output
        return actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
