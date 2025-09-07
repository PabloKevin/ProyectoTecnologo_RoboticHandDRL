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
                 learning_rate= 0.0002,
                 dropout2d=0.25, 
                 dropout=0.25, 
                 input_dims = (256, 256, 1), output_dims = 10,  
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
        self.features_maps = []  # To store feature maps

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
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            else:
                img = torch.tensor(img)

        img = img.float()  # conv2d espera float32

        # Asegurar 1 canal y orden NCHW
        if img.dim() == 2:
            # [H, W] -> [N=1, C=1, H, W]
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3:
            # Puede ser [H, W, C] o [C, H, W]
            if img.shape[0] in (1, 3):  # [C, H, W]
                img = img.unsqueeze(0)   # -> [N, C, H, W]
            else:                        # [H, W, C]
                img = img.permute(2, 0, 1).unsqueeze(0)  # -> [N, C, H, W]

        img = img.to(self.device, non_blocking=True)

        features_maps = []
        # Convolution block 1
        x = F.leaky_relu(self.conv1(img))
        features_maps.append(x)
        x = self.pool1(x)
        
        
        # Convolution block 2
        x = F.leaky_relu(self.conv2(x))
        features_maps.append(x)
        x = self.pool2(x)
        

        # Convolution block 3
        x = F.leaky_relu(self.conv3(x))
        features_maps.append(x)
        x = self.pool3(x)
        

        for fm in features_maps:
            self.features_maps.append(fm.cpu().detach().numpy())

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
    
    def load_model(self):
        state = torch.load(self.checkpoint_file, map_location=self.device)
        self.load_state_dict(state)
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
model_weight_dir = "Desarrollo/simulation/Env04/model_weights_docs/observer/v7/"
model_name = "observer_final_v7"

observer = ObserverNetwork(checkpoint_dir=model_weight_dir) # para ejecutar en vsc quitar el checkpoint para usar el que está por defecto. 
observer.checkpoint_file = os.path.join(observer.checkpoint_dir, model_name)


observer.load_model()
observer.eval()

for _ in range(1):
    img = get_random_image(train_dataset)
    observer(img)

    for fm in observer.features_maps:
        cols = int(np.sqrt(fm.shape[1]))
        cols = 8 if cols**2 != fm.shape[1] else cols
        print(cols)
        plot_feature_maps(fm.squeeze(0), ncols=cols)
    #plot_feature_maps(observer.features_maps[1][[0, 4, 6, 9, 22, 25, 30, 31], :, :], ncols=4)
    