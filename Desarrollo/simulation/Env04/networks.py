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
    def __init__(self, input_dims,  hidden_layers=[64,32,32], name='critic', checkpoint_dir='Desarrollo/simulation/Env04/tmp/td3', 
                 learning_rate=0.001):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims # check if this is necessary
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(input_dims, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.fc5 = nn.Linear(hidden_layers[3], 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Critic Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.fc1(input))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        q_value = self.fc5(x)

        return q_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


# Actor Network
class ActorNetwork(nn.Module):
    # Devuelve la acción a tomar en función del estado
    def __init__(self, input_dims=11, n_actions=1, hidden_layers=[64,32,32], name='actor', checkpoint_dir='Desarrollo/simulation/Env04/tmp/td3', learning_rate=0.001):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims # check if this is necessary
        self.checkpoint_dir = checkpoint_dir
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        
        self.fc1 = nn.Linear(input_dims, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.fc5 = nn.Linear(hidden_layers[3], n_actions)
        
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Actor Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        #print(f"Shape after conv3: {x.shape}")
        # Check if the input is a batch or a single image
        action = torch.tanh(self.fc5(x)) # output between (-1,1)
            
        return action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        print("Successfully loaded actor model")



# Observer Network
class ObserverNetwork(nn.Module):
    # Devuelve la acción a tomar en función del estado
    def __init__(self, 
                 conv_channels=[16, 32, 64], 
                 hidden_layers=[64, 32, 16], 
                 learning_rate= 0.0008,
                 dropout2d=0.2, 
                 dropout=0.25, 
                 input_dims = (256, 256, 1), output_dims = 10,  
                 name="observer_best_test_logits_2", checkpoint_dir='Desarrollo/simulation/Env04/tmp/observer'):
        super(ObserverNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.conv_channels = conv_channels
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout2d = dropout2d
        self.dropout = dropout
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=8, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=6, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=1, padding=2)
        
        # Pooling layers
        """self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)"""
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(input_dims[0] // 4, input_dims[1] // 4))
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=(input_dims[0] // 8, input_dims[1] // 8))
        self.pool3 = nn.AdaptiveMaxPool2d(output_size=(input_dims[0] // 16, input_dims[1] // 16)) 

        # Dropout2d para intentar mejorar el overfitting
        self.conv_dropout = nn.Dropout2d(p=dropout2d)

        # After three times pooling by factor of 2, 
        # the spatial dimensions become (H/8) x (W/8)
        self.fc1 = nn.Linear(conv_channels[2] * (input_dims[0] // 16) * (input_dims[1] // 16), hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], output_dims)
        
        self.dropout = nn.Dropout(p=dropout)        # Dropout en la parte fully-connected

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Created Observer Network on device: {self.device}")
        self.to(self.device)
        

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float).to(self.device)

        # Convolution block 1
        x = F.leaky_relu(self.conv1(img))
        x = self.pool1(x)
        
        # Convolution block 2
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)

        # Convolution block 3
        x = F.leaky_relu(self.conv3(x))
        x = self.pool3(x)

        # Apagar neuronas tras la 3ra capa conv
        x = self.conv_dropout(x)

        #print(f"Shape after conv3: {x.shape}")
        # Check if the input is a batch or a single image
        if len(x.shape) == 4:  # Batch case: [batch_size, channels, height, width]
            #x = x[:, [0, 4, 6, 9, 22, 25, 30, 31], :, :] # Just interesting features maps
            x = x.reshape((x.size(0), -1))  # Flatten each sample in the batch
        elif len(x.shape) == 3:  # Single image case: [channels, height, width]
            #x = x[[0, 4, 6, 9, 22, 25, 30, 31], :, :] # Just interesting features maps
            x = x.reshape(-1)  # Flatten the single image
        x = F.leaky_relu(self.fc1(x))

        # Apagar neuronas en fully-connected
        x = self.dropout(x)

        x = F.leaky_relu(self.fc2(x))

        x = self.dropout(x)

        x = F.leaky_relu(self.fc3(x))

        x = self.dropout(x)
        #tool_reg = F.leaky_relu(self.fc3(x))
        #tool_reg = torch.tanh(self.fc3(x))*3.5+2.5
        logits = self.fc4(x)
            
        return logits # Tool regresion

    def save_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            torch.save(self.state_dict(), checkpoint_file)
        

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        print("Successfully loaded observer model")