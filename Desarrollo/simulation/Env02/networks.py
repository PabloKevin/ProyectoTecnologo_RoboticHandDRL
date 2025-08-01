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
    def __init__(self, input_dims, n_actions, conv_channels=[16, 32, 64], hidden_size=256, name='critic', checkpoint_dir='Desarrollo/simulation/Env02/tmp/td3', 
                 learning_rate=0.001, n_choices_per_finger = 3):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims # check if this is necessary
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        self.n_choices_per_finger = n_choices_per_finger

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(conv_channels[2] * (input_dims[0] // 8) * (input_dims[1] // 8), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc3 = nn.Linear(1 + n_actions * n_choices_per_finger, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Critic Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, state, action_probs):
        state = state.to(self.device)
        action_probs = action_probs.reshape((action_probs.shape[0],-1)).to(self.device)
        x = F.leaky_relu(self.conv1(state))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        # Check if the input is a batch or a single image
        x = x.reshape((x.size(0), -1))  # Flatten each sample in the batch
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.cat([x, action_probs], dim=1)
        q_value = torch.clamp(self.fc3(x), max=5.0)

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
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc3 = nn.Linear(1, n_actions * n_choices_per_finger)
        
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Actor Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, state):
        state = state.to(self.device)
        x = F.leaky_relu(self.conv1(state))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        #print(f"Shape after conv3: {x.shape}")
        # Check if the input is a batch or a single image
        if len(x.shape) == 4:  # Batch case: [batch_size, channels, height, width]
            x = x.reshape((x.size(0), -1))  # Flatten each sample in the batch
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            logits = self.fc3(x).reshape((x.size(0),self.n_actions, self.n_choices_per_finger))
            # Use softmax with the logits so as to get probabilities
            probabilities = F.softmax(logits, dim=2)
            #actions = torch.argmax(probabilities, dim=2)  # Discrete action output

        elif len(x.shape) == 3:  # Single image case: [channels, height, width]
            x = x.reshape(-1)  # Flatten the single image
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            print(f"Observer: {x}")
            logits = F.leaky_relu(self.fc3(x)).reshape((self.n_actions, self.n_choices_per_finger))
            probabilities = F.softmax(logits, dim=1)
            #print("probabilities: ", probabilities)
            #actions = torch.argmax(probabilities, dim=1)  # Discrete action output
            
        return probabilities

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
