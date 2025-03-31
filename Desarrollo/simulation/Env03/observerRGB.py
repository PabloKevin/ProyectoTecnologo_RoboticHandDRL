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
from networks import ObserverNetwork
import time


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
    
    def __len__(self):
        return len(self.image_files)
    
    def __get_label_from_filename__(self, filename):
        """
        Convert a filename to a label based on its prefix.
        Customize this function as needed if your filenames differ.
        """
        filename_lower = filename.lower()
        if filename_lower.startswith("empty"):
            return -1.0
        elif filename_lower.startswith("tuerca"):
            return 0.0
        elif filename_lower.startswith("tornillo"):
            return 0.3
        elif filename_lower.startswith("clavo"):
            return 0.6
        elif filename_lower.startswith("lapicera"):
            return 10.0
        elif filename_lower.startswith("tenedor"):
            return 10.3
        elif filename_lower.startswith("cuchara"):
            return 10.6
        elif filename_lower.startswith("destornillador"):
            return 20.0
        elif filename_lower.startswith("martillo"):
            return 20.3
        elif filename_lower.startswith("pinza"):
            return 20.6
        else:
            # Default or unknown label
            return 99.9

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

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    print(train_dataset.image_files.head(5))
    print(val_dataset.image_files.head(5))
    print(test_dataset.image_files.head(5))

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

    # Now you can do, e.g., training with these loaders:
    # (Below is just an example snippet – adapt it to your ObserverNetwork code)

    
    conv_channels = [4, 8, 16]
    hidden_layers = [32, 8]
    learning_rate = 0.0008

    observer = ObserverNetwork(conv_channels=conv_channels, hidden_layers=hidden_layers, learning_rate=learning_rate)
    #observer.load_model()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Example training loop
    criterion = nn.MSELoss()
    n_epochs = 30

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

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
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
    
    print("Finished training!")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds = {(time.time() - start_time)/60:.2f} minutes")

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

    log_file.write(f'{run},-1,{avg_train_loss:.4f},-1,{avg_val_loss:.4f},-1,-1,{avg_test_loss:.4f},{test_duration:.2f},"{conv_channels}","{hidden_layers}",{learning_rate}\n')
