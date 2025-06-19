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
from sklearn.metrics import f1_score
from observerRGB import ObserverNetwork, MyImageDataset

def get_class_from_reg(reg, thresholds=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, float("inf")]):
        """
        Convert a regression value to a class label.
        """
        class_names = []
        for r in reg:
            for i, threshold in enumerate(thresholds):
                if r < threshold:
                    class_names.append(i)
                    break
        return class_names

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # For multiptocessing with CUDA

    # logs file
    log_path = "Desarrollo/simulation/Env04/"
    log_name = "observer_logs02.csv"
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

    batch_size = 64 #32
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader   = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=8)

    #observer = ObserverNetwork()
    observer = ObserverNetwork(checkpoint_dir='Desarrollo/simulation/Env04/model_weights_docs/observer/v5')
    #observer.checkpoint_file = os.path.join(observer.checkpoint_dir, "observer_epoch_100")
    #observer.load_model()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Example training loop
    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    criterion = nn.CrossEntropyLoss()

    n_epochs = 100
    best_test_f1 = 0.0
    start_time = time.time()
    for epoch in range(0,n_epochs):
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
        log_file.write(f"{run},{epoch+1},{avg_train_loss:.4f},{train_duration:.2f},{-1},{-1},{(train_duration):.2f},-1,-1,-1,-1,-1,-1\n")
        
        if (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == n_epochs - 1):
            observer.save_checkpoint(checkpoint_file=os.path.join(observer.checkpoint_dir, observer.name+'_epoch_'+str(epoch+1)))
            print(f"------\nCheckpoint saved for epoch {epoch + 1}\n------")
        
        
        if (epoch + 1) % 3 == 0 or epoch == n_epochs - 1:
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

                probs  = F.softmax(outputs_test, dim=1) 
                pred_labels = torch.argmax(probs, dim=1).cpu().numpy()
                #pred_labels = get_class_from_reg(outputs_test.cpu().numpy().argmax())
                f1 = f1_score(labels_test.cpu().numpy(), pred_labels, average="weighted")

                test_duration = (time.time() - start_test_time)
                print(f"           Test Loss: {avg_test_loss:.4f}")
                print(f"F1-score: {f1:.4f}")
                print(f"Test duration: {test_duration:.2f} seconds")
                
            log_file.write(f'{run},-1,{avg_train_loss:.4f},-1,-1,-1,-1,{avg_test_loss:.4f},{test_duration:.2f},"{observer.conv_channels}","{observer.hidden_layers}",{observer.learning_rate},{f1:.4f}\n')

            if f1 > best_test_f1:
                best_test_f1 = f1
                observer.save_checkpoint(checkpoint_file=os.path.join(observer.checkpoint_dir, observer.name+'_final_v1'))
                print(f"------\nBest F1-score checkpoint saved\n------")

    print("Finished training!")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds = {(time.time() - start_time)/60:.2f} minutes")

    