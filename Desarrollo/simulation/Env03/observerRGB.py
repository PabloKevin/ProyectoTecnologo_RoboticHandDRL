import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from SAM_pipe import Segmentator
import polars as pl
from networks import ObserverNetwork


class MyImageDataset(Dataset):
    """
    Custom Dataset that loads images from a directory and applies your
    DataSet_editor transforms. It uses a labeling convention based on
    filename prefixes.
    """
    def __init__(self, image_dir, segmentator, extensions=(".png"), name="names"):
        """
        Args:
            image_dir (str): Directory containing all the images.
            image_shape (tuple): (width, height) or (height, width).
            extensions (tuple): Valid image extensions to read from disk.
        """
        super().__init__()
        self.image_dir = image_dir
        if segmentator is None:
            self.segmentator 
        else:
            self.segmentator = segmentator
        self.image_shape = segmentator.output_dims
        
        # Gather all valid image paths
        self.image_files = []
        for f in os.listdir(image_dir):
            if f.lower().endswith(extensions):
                self.image_files.append(f)
        self.image_files = pl.Series(name, self.image_files)
        # Sort or shuffle if desired
        #random.shuffle(self.image_files)
    
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
        
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")

        # Get label from the filename
        label = self.__get_label_from_filename__(filename)

        # Pasar imagen por el segmentator
        bw_mask = self.segmentator.predict(img, render=False)
        bw_mask = np.expand_dims(bw_mask, axis=0)
        
        # Convert everything into torch tensors
        img_tensor = torch.tensor(bw_mask, dtype=torch.float)  # (1, H, W)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return img_tensor, label_tensor


if __name__ == "__main__":
    train_dataset_dir = "Desarrollo/simulation/Env03/DataSets/TrainSet"
    test_dataset_dir = "Desarrollo/simulation/Env03/DataSets/TestSet"
    checkpoint_dir="Desarrollo/simulation/Env03/models_params_weights/"
    segmentator = Segmentator(checkpoint_dir=checkpoint_dir+"SAM/sam_vit_b_01ec64.pth")
    # Create the full dataset
    full_train_dataset = MyImageDataset(train_dataset_dir, segmentator, name="full_train_dataset")
    test_dataset = MyImageDataset(test_dataset_dir, segmentator, name="test_dataset")
    
    # For example, make an 80/20 split for train/validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size   = len(full_train_dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_files, val_files = random_split(full_train_dataset.image_files, [train_size, val_size], generator=generator1)
    train_dataset, val_dataset = MyImageDataset(train_dataset_dir, segmentator), MyImageDataset(train_dataset_dir, segmentator)
    train_dataset.image_files = pl.Series("train_dataset", list(train_files))
    val_dataset.image_files = pl.Series("val_dataset", list(val_files))

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    print(train_dataset.image_files.head(5))
    print(val_dataset.image_files.head(5))
    print(test_dataset.image_files.head(5))

    """
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Now you can do, e.g., training with these loaders:
    # (Below is just an example snippet – adapt it to your ObserverNetwork code)

    
    observer = ObserverNetwork()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Example training loop
    criterion = nn.MSELoss()
    n_epochs = 2

    for epoch in range(n_epochs):
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
        print(f"Epoch [{epoch+1}/{n_epochs}] - Train Loss: {avg_train_loss:.4f}")

        # Optionally validate
        observer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_val, labels_val in val_loader:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                outputs_val = observer(images_val).squeeze()
                loss_val = criterion(outputs_val, labels_val)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"           Validation Loss: {avg_val_loss:.4f}")

    print("Finished training!")
