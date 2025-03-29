import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import cv2
from DataSet_editor import DataSet_editor
import matplotlib.pyplot as plt
from SAM_pipe import Segmentator


class DynamicBatchGenerator:
    def __init__(self, batch_size, image_shape=(256,256,1)):
        self.batch_size = batch_size
        self.image_shape = image_shape

    def get_batch(self):
        """
        Generate a batch of images and labels dynamically.
        :return: Batch of images and corresponding labels.
        """
        images = []
        labels = []
        for _ in range(self.batch_size):
            # Dynamically generate an image using get_initial_image()
            image = self.get_image()
            label = self.get_label(image)

            # Normalize the image and add a channel dimension
            #image = np.expand_dims(image / 255.0, axis=0)
            image = np.expand_dims(image, axis=0)

            images.append(image)
            labels.append(label)

        # Convert to numpy array for being faster
        images = np.array(images)
        labels = np.array(labels)

        # Convert to PyTorch tensors
        images = torch.tensor(images, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)

        return images, labels
    
    def get_image(self):
        # Directory containing images
        image_dir = "Desarrollo/simulation/Env03/DataSets/B&W_Tools/"
        
        # List all image files in the directory
        #image_files = [f for f in os.listdir(image_dir)]
        # ["bw_Martillo01.jpg", "empty.png", "bw_Lapicera01.png", "bw_destornillador01.jpg", "bw_tornillo01.jpg"] ["bw_lapicera02.png","bw_lapicera02.png"]
        images_of_interest = ["bw_Martillo01.jpg", "empty.png", "bw_Lapicera01.png", "bw_destornillador01.jpg", "bw_tornillo01.jpg", "bw_lapicera02.png"]
        image_files = [f for f in os.listdir(image_dir) if f in images_of_interest]
        # Check how many images are there
        num_images = len(image_files)
        
        # Create a random index
        random_index = np.random.randint(0, num_images)
        
        # Select an image with that index
        selected_image_path = os.path.join(image_dir, image_files[random_index])
        #selected_image_path = "Desarrollo/simulation/Env03/DataSets/B&W_Tools/bw_Martillo01.jpg"
        
        # Load the image
        img = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
        # Transform the image so there is a "different" image for each episode
        editor = DataSet_editor(img_width=self.image_shape[0], img_height=self.image_shape[1])
        img = editor.transform_image(img)
        #white_pixels = np.argwhere(img == 255)
        #print(len(white_pixels))
        # Convert 255 pixels to 1
        img[img < 255/2] = 0  
        img[img >=  255/2] = 1
        #file = "Desarrollo/simulation/Env01/img.txt"
        #np.savetxt(file, img, fmt="%d", delimiter=" ") 
        #img = np.expand_dims(img, axis=-1)
        return img
    
    def get_label(self, image_name):
        if image_name.startswith("empty"):
            return -1.0
        elif image_name.startswith("tuerca"):
            return 0.0
        elif image_name.startswith("tornillo"):
            return 0.3
        elif image_name.startswith("clavo"):
            return 0.6
        elif image_name.startswith("lapicera"):
            return 10.0
        if image_name.startswith("tenedor"):
            return 10.3
        elif image_name.startswith("cuchara"):
            return 10.6
        elif image_name.startswith("destornillador"):
            return 20.0
        elif image_name.startswith("martillo"):
            return 20.3
        elif image_name.startswith("pinza"):
            return 20.6


# Observer Network
class ObserverNetwork(nn.Module):
    # Devuelve la acción a tomar en función del estado
    def __init__(self, input_dims, output_dims = 1, conv_channels=[16, 32, 64], hidden_layers=[64,8], name='observer', checkpoint_dir='Desarrollo/simulation/Env03/tmp/observer', learning_rate=0.001):
        super(ObserverNetwork, self).__init__()
        self.input_dims = input_dims
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
        img = img.to(self.device)
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

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        print("Successfully loaded observer model")


class Trainer():
    def __init__(self, batch_size = 32, image_shape = (256, 256, 1), epochs = 20, batches_per_epoch = 100, learning_rate = 0.001, checkpoint_dir="Desarrollo/simulation/Env03/models_params_weights/"):
        # Parameters
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.learning_rate = learning_rate
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        # Initialize Segmentator Network
        self.segmentator = Segmentator(checkpoint_dir=checkpoint_dir+"SAM/sam_vit_b_01ec64.pth")

        # Initialize Observer Network
        self.observer = ObserverNetwork(input_dims=self.image_shape, learning_rate=self.learning_rate)
        self.observer.to(self.device)


    def train(self, epochs = None):
        if epochs is None:
            epochs = self.epochs
        device = self.device
        # Initialize Dynamic Batch Generator
        dynamic_batch_generator = DynamicBatchGenerator(batch_size=self.batch_size, image_shape=self.image_shape)

        observer = self.observer

        # Loss function and optimizer
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        # Training loop
        for epoch in range(epochs, ):
            observer.train()
            total_loss = 0

            for batch_idx in range(self.batches_per_epoch):  # Assuming 100 batches per epoch
                images, labels = dynamic_batch_generator.get_batch()
                images, labels = images.to(device), labels.to(device)

                observer.optimizer.zero_grad()
                outputs = observer(images).squeeze()  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                observer.optimizer.step()  # Update weights

                total_loss += loss.item()

            avg_loss = total_loss / self.batches_per_epoch  # Average loss for the epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # Save model checkpoint periodically
            if (epoch + 1) % 5 == 0:
                observer.save_checkpoint()
                print(f"Checkpoint saved for epoch {epoch + 1}")

        print("Training complete!")

class Evaluator():
    def __init__(self, observer):
        self.observer = observer

    def render(self, img, timeout=None):
        predicted_tool = self.observer(img).cpu().detach().numpy()
        img = img.squeeze(0).cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title('Episode Image')
        plt.axis('off')  # Ocultar los ejes
        
        # Add text annotation for finger states
        text = f"Predicted_tool: {predicted_tool}"
        plt.text(-12, 266, text, color='white', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.7))
        
        if timeout is not None:
            plt.show(block=False)
            plt.pause(timeout)  # Show plot for 0.5 seconds
            plt.close()  # Close the plot window
        else:
            plt.show()

if __name__ == '__main__':
    
    observer = ObserverNetwork(input_dims=(256, 256, 1))
    observer.load_checkpoint()

    images, labels = DynamicBatchGenerator(batch_size=10).get_batch()
    evaluator = Evaluator(observer)

    for img in images:
        evaluator.render(img)
    
    
    

    #trainer = Trainer()
    #trainer.train()