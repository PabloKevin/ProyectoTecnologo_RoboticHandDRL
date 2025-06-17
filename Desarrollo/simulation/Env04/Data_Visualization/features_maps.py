import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Ruta del archivo actual
current_dir = os.path.dirname(__file__)
# Un nivel arriba (donde está networks.py)
parent_dir = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(parent_dir))

from observerRGB import MyImageDataset

# Observer Network
class ObserverNetwork(nn.Module):
    def __init__(self, 
                 conv_channels=[16, 32, 64], 
                 hidden_layers=[64, 32, 16], 
                 learning_rate= 0.0008,
                 dropout2d=0.3, 
                 dropout=0.3, 
                 input_dims = (256, 256, 1), output_dims = 1, 
                 name='observer', checkpoint_dir='Desarrollo/simulation/Env04/tmp/observer'):
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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_supervised')
        self.features_maps = [None, None, None]

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=1, padding=2)
        
        # Pooling layers
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(input_dims[0] // 2, input_dims[1] // 2))
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(input_dims[0] // 4, input_dims[1] // 4))

        # Dropout2d para intentar mejorar el overfitting
        self.conv_dropout = nn.Dropout2d(p=dropout2d)

        # After three times pooling by factor of 2, 
        # the spatial dimensions become (H/8) x (W/8)
        self.fc1 = nn.Linear(4 * (input_dims[0] // 4) * (input_dims[1] // 4), hidden_layers[0])
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

        img = torch.tensor(img, dtype=torch.float).to(self.device)
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = self.pool1(x)
        features_maps_1 = x
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool2(x)
        features_maps_2 = x
        
        self.features_maps[0] = features_maps_1.cpu().detach().numpy()
        self.features_maps[1] = features_maps_2.cpu().detach().numpy()

        # Apagar neuronas tras la 3ra capa conv
        x = self.conv_dropout(x)

        #print(f"Shape after conv3: {x.shape}")
        # Check if the input is a batch or a single image
        if len(x.shape) == 4:  # Batch case: [batch_size, channels, height, width]
            #x = x[:, [0, 4, 6, 9, 22, 25, 30, 31], :, :]
            x = x.reshape((x.size(0), -1))  # Flatten each sample in the batch
        elif len(x.shape) == 3:  # Single image case: [channels, height, width]
            #x = x[[0, 4, 6, 9, 22, 25, 30, 31], :, :]
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
        tool_reg = self.fc4(x)
            
        return tool_reg # Tool regresion
    
    #def save_checkpoint(self):
    #    torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        print("Successfully loaded observer model")


def plot_feature_maps(feature_maps, ncols=4, cmap='gray'):
    """
    feature_maps: torch.Tensor of shape (C, H, W)
    ncols: number of columns in the subplot
    cmap: colormap to use for imshow
    """
    n_filters = feature_maps.shape[0]
    nrows = (n_filters + ncols - 1) // ncols  # round up

    fig, axarr = plt.subplots(nrows, ncols, figsize=(12, 12))
    axarr = axarr.flat if nrows > 1 else [axarr]  # handle 1D vs 2D array of axes

    for i in range(n_filters):
        # Convert each channel to CPU numpy if on GPU
        axarr[i].imshow(feature_maps[i], cmap=cmap)
        axarr[i].axis('off')

    # Hide unused subplots if any
    for j in range(i+1, nrows*ncols):
        axarr[j].axis('off')

    plt.tight_layout()
    plt.show()

def get_random_image(train_dataset):
    idx = np.random.randint(0, len(train_dataset.image_files))
        
    return train_dataset.__getitem__(idx)[0]
        
train_dataset = MyImageDataset("Desarrollo/simulation/Env04/DataSets/TrainSet_masks", name="full_train_masks_dataset")

#observer = ObserverNetwork(checkpoint_dir="Desarrollo/simulation/Env04/tmp/observer") # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
#observer.checkpoint_file = os.path.join(observer.checkpoint_dir, "observer_best_test")
observer = ObserverNetwork(checkpoint_dir="Desarrollo/simulation/Env04/models_params_weights/observer") # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
observer.checkpoint_file = os.path.join(observer.checkpoint_dir, "observer_best_test_logits_best2")
observer.load_model()
observer.eval()

for _ in range(1):
    img = get_random_image(train_dataset)
    observer(img)

    plot_feature_maps(observer.features_maps[0], ncols=1)
    print("shape:", observer.features_maps[1].shape)
    plot_feature_maps(observer.features_maps[1], ncols=2)
    #plot_feature_maps(observer.features_maps[1][[0, 4, 6, 9, 22, 25, 30, 31], :, :], ncols=4)
    