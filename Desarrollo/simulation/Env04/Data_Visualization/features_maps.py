import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from DataSet_editor import DataSet_editor

# Observer Network
class ObserverNetwork(nn.Module):
    # Devuelve la acción a tomar en función del estado
    def __init__(self, input_dims = (256, 256, 1), output_dims = 1, conv_channels=[16, 32, 64], hidden_layers=[64,8], name='observer', checkpoint_dir='Desarrollo/simulation/Env04/tmp/observer', learning_rate=0.001):
        super(ObserverNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_supervised')
        self.features_maps = [None, None, None]

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
        features_maps_1 = self.conv1(img)
        x = F.leaky_relu(features_maps_1)
        features_maps_2 = self.conv2(x)
        x = F.leaky_relu(features_maps_2)
        features_maps_3 = self.conv3(x)
        x = F.leaky_relu(features_maps_3)

        
        self.features_maps[0] = features_maps_1.cpu().detach().numpy()
        self.features_maps[1] = features_maps_2.cpu().detach().numpy()
        self.features_maps[2] = features_maps_3.cpu().detach().numpy()
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

def get_random_image():
    image_dir = "./DataSets/B&W_Tools"
    tools_of_interest = [f for f in os.listdir(image_dir)]
    num_images = len(tools_of_interest)
    random_index = np.random.randint(0, num_images)
    selected_image_path = os.path.join(image_dir, tools_of_interest[random_index])

    img = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)

    editor = DataSet_editor()
    img = editor.transform_image(img)

    img[img < 255/2] = 0  
    img[img >=  255/2] = 1
    img = np.expand_dims(img, axis=0)
    return img

observer = ObserverNetwork(checkpoint_dir="./tmp/observer") # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
observer.load_model()

for _ in range(1):
    img = get_random_image()
    observer(img)

    plot_feature_maps(observer.features_maps[0], ncols=4)
    plot_feature_maps(observer.features_maps[1], ncols=8)
    plot_feature_maps(observer.features_maps[2], ncols=16)