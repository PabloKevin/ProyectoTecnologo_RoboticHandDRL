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
from observerRGB import ObserverNetwork, MyImageDataset


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # For multiptocessing with CUDA

    # logs file
    log_path = "Desarrollo/simulation/Env04/"
    log_name = "observer_logs.csv"
    log_file = open(log_path+log_name, "a")

    log_df = pl.read_csv(log_path+log_name)
    if log_df.is_empty():
        run = 0
    else:
        run = log_df.select("run")[-2].item() + 1


    train_dataset_dir = "Desarrollo/simulation/Env04/DataSets/TrainSet_masks"
    test_dataset_dir = "Desarrollo/simulation/Env04/DataSets/TestSet_masks"
    # Create the full dataset
    full_train_dataset = MyImageDataset(train_dataset_dir, name="full_train_masks_dataset")
    test_dataset = MyImageDataset(test_dataset_dir, name="test_masks_dataset")

    # Create DataLoaders for batching
    # example: batch_size=32

    batch_size = 128
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader   = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=8)

    observer = ObserverNetwork()
    observer.checkpoint_file = os.path.join(observer.checkpoint_dir, "observer_best_test")
    observer.load_model()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Example training loop
    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    criterion = nn.CrossEntropyLoss()
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

        """# Validate
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
        """
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
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

            log_file.write(f'{run},-1,{avg_train_loss:.4f},-1,-1,-1,-1,{avg_test_loss:.4f},{test_duration:.2f},"{observer.conv_channels}","{observer.hidden_layers}",{observer.learning_rate}\n')

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                observer.save_checkpoint(checkpoint_file=os.path.join(observer.checkpoint_dir, observer.name+'_best_test'))
                print(f"------\nBest test loss checkpoint saved\n------")

    print("Finished training!")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds = {(time.time() - start_time)/60:.2f} minutes")

    