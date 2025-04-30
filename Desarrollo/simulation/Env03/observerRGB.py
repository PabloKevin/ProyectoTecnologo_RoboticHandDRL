import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import polars as pl
import time

# Observer Network
class ObserverNetwork(nn.Module):
    # Devuelve la acción a tomar en función del estado
    def __init__(self, 
                 conv_channels=[16, 32, 64], 
                 hidden_layers=[32, 16, 8], 
                 learning_rate= 0.0008,
                 dropout2d=0.1, 
                 dropout=0.1, 
                 input_dims = (256, 256, 1), output_dims = 1, 
                 name='observer', checkpoint_dir='Desarrollo/simulation/Env03/tmp/observer'):
        super(ObserverNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_supervised')

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=1, padding=2)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout2d para intentar mejorar el overfitting
        self.conv_dropout = nn.Dropout2d(p=dropout2d)

        # After three times pooling by factor of 2, 
        # the spatial dimensions become (H/8) x (W/8)
        self.fc1 = nn.Linear(conv_channels[2] * (input_dims[0] // 8) * (input_dims[1] // 8), hidden_layers[0])
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
            x = x.reshape((x.size(0), -1))  # Flatten each sample in the batch
        elif len(x.shape) == 3:  # Single image case: [channels, height, width]
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

    def save_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            torch.save(self.state_dict(), checkpoint_file)
        

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        print("Successfully loaded observer model")


class MyImageDataset(Dataset):
    """
    Custom Dataset that loads images from a directory and applies your
    DataSet_editor transforms. It uses a labeling convention based on
    filename prefixes.
    """
    def __init__(self, image_dir, extensions=(".png"), name="names"):
        """
        Args:
            image_dir (str): Directory containing all the images.
            image_shape (tuple): (width, height) or (height, width).
            extensions (tuple): Valid image extensions to read from disk.
            name (str): Name of the dataset.
        """
        super().__init__()
        self.image_dir = image_dir
        self.image_shape = (256, 256) 
        
        # Gather all valid image paths
        self.image_files = []
        for f in os.listdir(image_dir):
            if f.lower().endswith(extensions):
                self.image_files.append(f)
        self.image_files = pl.Series(name, self.image_files)
        self.label_mapping = {
                                "empty": 0.0,
                                "tuerca": 1.0,
                                "tornillo": 1.3,
                                "clavo": 1.6,
                                "lapicera": 2.6,
                                "tenedor": 2.9,
                                "cuchara": 3.2,
                                "destornillador": 4.2,
                                "martillo": 4.5,
                                "pinza": 4.8,
                                "default": -1.0
                            }

    
    def __len__(self):
        return len(self.image_files)
    
    def __get_label_from_filename__(self, filename):
        """
        Convert a filename to a label based on its prefix.
        Customize this function as needed if your filenames differ.
        """
        filename_lower = filename.lower()
        for name, label in self.label_mapping.items():
            if filename_lower.startswith(name):
                return label
        

    def __getitem__(self, idx):
        """
        Loads an image 
        and returns (image_tensor, label_tensor).
        """
        filename = self.image_files[idx]
        file_path = os.path.join(self.image_dir, filename)
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")

        # Get label from the filename
        label = self.__get_label_from_filename__(filename)

        img = np.expand_dims(img, axis=0)
        
        # Convert everything into torch tensors
        img_tensor = torch.tensor(img, dtype=torch.float)  # (1, H, W)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return img_tensor, label_tensor


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # For multiptocessing with CUDA

    # logs file
    log_path = "Desarrollo/simulation/Env03/"
    log_name = "observer_logs.csv"
    log_file = open(log_path+log_name, "a")

    log_df = pl.read_csv(log_path+log_name)
    if log_df.is_empty():
        run = 0
    else:
        run = log_df.select("run")[-2].item() + 1


    train_dataset_dir = "Desarrollo/simulation/Env03/DataSets/TrainSet_masks"
    test_dataset_dir = "Desarrollo/simulation/Env03/DataSets/TestSet_masks"
    # Create the full dataset
    full_train_dataset = MyImageDataset(train_dataset_dir, name="full_train_masks_dataset")
    test_dataset = MyImageDataset(test_dataset_dir, name="test_masks_dataset")
    
    # For example, make an 80/20 split for train/validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size   = len(full_train_dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_files, val_files = random_split(full_train_dataset.image_files, [train_size, val_size], generator=generator1)
    train_dataset, val_dataset = MyImageDataset(train_dataset_dir), MyImageDataset(train_dataset_dir)
    train_dataset.image_files = pl.Series("train_masks_dataset", list(train_files))
    val_dataset.image_files = pl.Series("val_masks_dataset", list(val_files))

    """print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    print(train_dataset.image_files.head(5))
    print(val_dataset.image_files.head(5))
    print(test_dataset.image_files.head(5))"""

    """    # Optional: Visualize a sample image and its label
    img1 = train_dataset[1][0]
    print("label,", train_dataset[1][1])
    from matplotlib import pyplot as plt
    plt.imshow(img1.squeeze(), cmap='gray')
    plt.title('Black and White Mask')
    plt.axis('off') 
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    """

    # Create DataLoaders for batching
    # example: batch_size=32

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader   = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=8)

    observer = ObserverNetwork()
    #observer.load_model()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Example training loop
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    n_epochs = 100

    best_val_loss = float('inf')
    best_test_loss = float('inf')
    start_time = time.time()
    for epoch in range(n_epochs):
        start_epoch_time = time.time()
        observer.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            observer.optimizer.zero_grad()
            outputs = observer(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            observer.optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        end_train_time = time.time()
        train_duration = (end_train_time - start_epoch_time)  # in seconds
        print(f"Epoch [{epoch+1}/{n_epochs}] - Train Loss: {avg_train_loss:.4f}")
        print(f"Train duration: {train_duration:.2f} seconds")

        if ((epoch + 1) % 5 == 0 or epoch == n_epochs - 1) and avg_val_loss < best_val_loss:
            observer.save_checkpoint()
            print(f"------\nCheckpoint saved for epoch {epoch + 1}\n------")

        # Validate
        start_val_time = time.time()
        observer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_val, labels_val in val_loader:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                outputs_val = observer(images_val).squeeze()
                loss_val = criterion(outputs_val, labels_val)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader)
        end_val_time = time.time()
        val_duration = (end_val_time - start_val_time)
        print(f"           Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation duration: {val_duration:.2f} seconds")
        print(f"Epoch duration: {(train_duration + val_duration):.2f} seconds\n")
        log_file.write(f"{run},{epoch+1},{avg_train_loss:.4f},{train_duration:.2f},{avg_val_loss:.4f},{val_duration:.2f},{(train_duration + val_duration):.2f},-1,-1,-1,-1,-1\n")
        
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            observer.eval()
            start_test_time = time.time()
            test_loss = 0.0
            with torch.no_grad():
                for images_test, labels_test in test_loader:
                    images_test, labels_test = images_test.to(device), labels_test.to(device)
                    outputs_test = observer(images_test).squeeze()
                    loss_test = criterion(outputs_test, labels_test)
                    test_loss += loss_test.item()
                avg_test_loss = test_loss / len(test_loader)

                test_duration = (time.time() - start_test_time)
                print(f"           Test Loss: {avg_test_loss:.4f}")
                print(f"Test duration: {test_duration:.2f} seconds")

            log_file.write(f'{run},-1,{avg_train_loss:.4f},-1,{avg_val_loss:.4f},-1,-1,{avg_test_loss:.4f},{test_duration:.2f},"{observer.conv_channels}","{observer.hidden_layers}",{observer.learning_rate}\n')

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                observer.save_checkpoint(checkpoint_file=os.path.join(observer.checkpoint_dir, observer.name+'_best_test'))
                print(f"------\nBest test loss checkpoint saved\n------")

    print("Finished training!")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds = {(time.time() - start_time)/60:.2f} minutes")

    