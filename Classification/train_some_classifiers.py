# tmux new -s train_session
# activate the environment
# python train_some_classifiers.py.py
# Ctrl + B, then D
# tmux attach -t train_session
# tmux kill-session -t train_session
import numpy as np
import os
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from natsort import natsorted
import torch
import pickle
import json
from monai.networks.nets import DenseNet121, resnet18
import torch.nn as nn
from Classification.classifier_helper import ClassificatorBlobHelper
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise,
    EnsureType, EnsureChannelFirst
)
from Classification.classifier_helper import ObjectPatchDataset, calculate_test_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

imgs_20xRenamed = []
with open('save_data/3D_images_Renamed/imgs_20xRenamed.pkl', 'rb') as f:
    imgs_20xRenamed = pickle.load(f)
masks_20xRenamed = []
with open('save_data/masks/20xRenamed/masks3D_CELLPOSE_RUN_1.pkl', 'rb') as f:
    masks_20xRenamed = pickle.load(f)
gt_classes = {}
with open("classes_0.json", "r") as f:
    gt_classes = json.load(f)

image_index = 0 # potentially add more images as a list
label_indices = [int(i) for i in np.unique(masks_20xRenamed[image_index]) if i != 0]

#for image_index in range(len(masks_20xRenamed)):
#    all_indices = [(image_index, i) for i in label_indices]
all_indices = [(0, i) for i in label_indices]   
all_labels = [gt_classes[str(i)] for i in label_indices]

train_idx, val_idx, train_lbls, val_lbls = train_test_split(all_indices, all_labels, test_size=0.2, random_state=42)

densenet_model_3 = DenseNet121(spatial_dims=3, in_channels=3, out_channels=5) 
densenet_model_4 = DenseNet121(spatial_dims=3, in_channels=4, out_channels=5) 

resnet_model_3 = resnet18(spatial_dims=3, n_input_channels=3, num_classes=5)
resnet_model_4 = resnet18(spatial_dims=3, n_input_channels=4, num_classes=5)
model = densenet_model_3
blb = ClassificatorBlobHelper()

train_transforms = Compose([
    RandRotate90(prob=0.5, spatial_axes=(0, 1)),  # rotate in XY
    RandFlip(prob=0.5, spatial_axis=0),           # flip on Z
    RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),
    EnsureType()  # Makes sure it's a torch.Tensor
])

train_dataset = ObjectPatchDataset(blb, train_idx, train_lbls, transform=train_transforms)
val_dataset = ObjectPatchDataset(blb, val_idx, val_lbls)

for vers in range(1, 7):
    #version control
    VERSION = vers
    if vers == 1:
        #in_channel_dist=True and in_channels=3 - DenseNet
        train_dataset.change_version(1)
        val_dataset.change_version(1)
    if vers == 2:
        #in_channel_dist=False and in_channels=4 - DenseNet
        train_dataset.change_version(2)
        val_dataset.change_version(2)
        model = densenet_model_4
    if vers == 3:
        #in_channel_dist=True and in_channels=3 - ResNet
        train_dataset.change_version(1)
        val_dataset.change_version(1)
        model = resnet_model_3
    if vers == 4:
        #in_channel_dist=False and in_channels=4 - ResNet
        train_dataset.change_version(2)
        val_dataset.change_version(2)
        model = resnet_model_4
    if vers == 5:
        # binary Mask in_channels=3 - DenseNet
        train_dataset.change_version(3)
        val_dataset.change_version(3)
        model = densenet_model_3
    if vers == 6:
        #binary Mask and in_channels=3 - ResNet
        train_dataset.change_version(3)
        val_dataset.change_version(3)
        model = resnet_model_3
    
    print("-"*100)
    print(f"Training version {VERSION} with {len(train_dataset)} training samples and {len(val_dataset)} validation samples. Size of each picture: {train_dataset[0][0].shape}")
    print("-"*100)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    history = {"loss": [], "val_loss": [], "val_acc": []}

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_dir = "save_data/checkpoints"

    num_epochs = 75

    best_val_acc = float('-inf')  

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Debug shapes
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch}, Batch {batch_idx} of {len(train_loader)}")

            # Forward
            pred = model(X)
            loss = criterion(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss, val_acc = calculate_test_loss(model, val_loader, criterion, device)
        avg_loss = running_loss / len(train_loader)
        print("-" * 50)
        print(f"[Epoch {epoch+1}/{num_epochs}]: Avg_loss = {avg_loss}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        print("-" * 50)
        history["loss"].append(avg_loss)
        history["val_loss"].append(val_loss)    
        history["val_acc"].append(val_acc)    
        #print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # Save model checkpoint (overwrite to save space)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_v{VERSION}.pt")
        history_path = os.path.join(checkpoint_dir, f"history_v{VERSION}.json")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
        with open(history_path, "w") as f:
            json.dump(history, f)
        print(f"Saved checkpoint: {checkpoint_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc  
            best_model_path = os.path.join(checkpoint_dir, f"best_model_v{VERSION}.pt")
            
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_acc": val_acc,
            }, best_model_path)
            
            print(f"New best model saved with val_acc = {val_acc:.4f} at: {best_model_path}")