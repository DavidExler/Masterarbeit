# tmux new -s train_session
# activate the environment
# python train_some_classifiers.py
# Ctrl + B, then D
# tmux attach -t train_session
# tmux kill-session -t train_session
import numpy as np
import os
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
#from natsort import natsorted
import torch
import pickle
import json
#from monai.networks.nets import DenseNet121, resnet18
import torch.nn as nn
from classifier_helper import ClassificatorBlobHelper, Hybrid3Dto2D
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise,
    EnsureType, EnsureChannelFirst
)
from classifier_helper import ObjectPatchDataset, calculate_test_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from classifier_helper import SAMClassifier3D, SAMClassifierMLP, SAMClassifier3D_CENTER_AWARE, get_resnet18_encoder, get_swinl_encoder, get_convnextxl_encoder, get_efficientnetv2l_encoder, get_resnet101_encoder, switch_model
from cellpose import models
import time
from collections import Counter
#
# NOTE: Vielleicht noch Z_achse beschraenken, jetzt sind immer alle slices drin
#

imgs_20xRenamed = []
with open('data/imgs_FINAL.pkl', 'rb') as f:
    imgs_20xRenamed = pickle.load(f)
masks_20xRenamed = []
with open('data/masks_FINAL.pkl', 'rb') as f:
    masks_20xRenamed = pickle.load(f)

with open("data/image_blob_indices_FINAL.pkl", "rb") as f:
    all_indices = pickle.load(f)

with open("data/labels_FINAL.pkl", "rb") as f:
    all_labels = pickle.load(f)

train_idx, val_idx, train_lbls, val_lbls = train_test_split(
    all_indices,
    all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels  # optional: ensures label distribution is balanced
)
for (img_idx, blob_idx), label in zip(train_idx[:5], train_lbls[:5]):
    print(f"Image {img_idx}, Blob {blob_idx} → Class {label}")


#pathology_model = DenseNet121(spatial_dims=2, in_channels=4, out_channels=4)
#checkpoint = torch.load("save_data/checkpoints/pathology.pt", map_location="cpu")
#pathology_model.load_state_dict(checkpoint)
#pathology_model.class_layers.out = nn.Linear(in_features=1024, out_features=5)
#pathology_model = Hybrid3Dto2D(pathology_model)
#
#for name, param in pathology_model.named_parameters():
#    if not (
#        "features.denseblock4.denselayer15" in name
#        or "features.denseblock4.denselayer16" in name
#        or "features.norm5" in name
#        or "class_layers.out" in name
#    ):
#        param.requires_grad = False
#        print(f"Freezing parameter: {name}")
#
#
#densenet_model_3 = DenseNet121(spatial_dims=3, in_channels=3, out_channels=5) 
#densenet_model_4 = DenseNet121(spatial_dims=3, in_channels=4, out_channels=5) 
#
#resnet_model_3 = resnet18(spatial_dims=3, n_input_channels=3, num_classes=5)
#resnet_model_4 = resnet18(spatial_dims=3, n_input_channels=4, num_classes=5)
#model = densenet_model_3

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

model = models.CellposeModel(gpu=False)
enc = model.net.encoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#enc.to(device)
batch_size = 32


blb = ClassificatorBlobHelper()

train_transforms = Compose([
    RandRotate90(prob=0.5, spatial_axes=(0, 1)),  # rotate in XY
    RandFlip(prob=0.5, spatial_axis=0),           # flip on Z
    RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),
    EnsureType()  # Makes sure it's a torch.Tensor
])

train_dataset = ObjectPatchDataset(blb, train_idx, train_lbls, transform=train_transforms)
val_dataset = ObjectPatchDataset(blb, val_idx, val_lbls)

for vers in range(26, 33):
    if vers % 2 != 0:
        continue  #Aktuell nur 3D Klassifikator, ohne Vortraining!
    VERSION = vers

    model, batch_size, train_loader, val_loader = switch_model(vers, train_dataset, val_dataset, device)
    print(f"Number of learnable parameters: {len([p for p in model.parameters() if p.requires_grad])}")
    print("-"*100)
    print(f"Training version {VERSION} with {len(train_dataset)} training samples and {len(val_dataset)} validation samples. Size of each picture: {train_dataset[0][0].shape}")
    print("-"*100)
    
    
    history = {"loss": [], "val_loss": [], "val_acc": []}


    #if vers == 12:
    #    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #    
    #    if torch.cuda.device_count() > 1:
    #        print(f"Using {torch.cuda.device_count()} GPUs!")
    #        model = nn.DataParallel(model)  # This wraps the model for multi-GPU use
    #else:
    #    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
    #if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs!")
    #    model = nn.DataParallel(model)
    model = model.to(device)
 

    checkpoint_dir = "data/checkpoints/inference/Transfer_from_semi"

    num_epochs = 1

    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_v{VERSION}.pt")
    val_acc_path = os.path.join(checkpoint_dir, f"val_acc_v{VERSION}.json")
    predictions_path = os.path.join(checkpoint_dir, f"predictions_v{VERSION}.json")

    if os.path.exists(checkpoint_path):
        print(f"test loss for checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Best epoch was {epoch}")
    else:
        print("no model to eval")
        break

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, predicted, labels = calculate_test_loss(model, val_loader, criterion, device) 
    val_acc_json = {"val_loss": val_loss, "val_acc": val_acc}
    predicted = [int(x) for x in predicted]
    labels = [int(x) for x in labels]


    with open(val_acc_path, "w") as f:
        json.dump(val_acc_json, f)
    with open(predictions_path, "w") as f:
        json.dump({"predicted": predicted, "labels": labels}, f)