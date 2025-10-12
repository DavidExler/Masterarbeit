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
from helpers.classifier_helper import ClassificatorBlobHelper, Hybrid3Dto2D, read_new_blob_folder
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise,
    EnsureType, EnsureChannelFirst
)
from helpers.classifier_helper import ObjectPatchDataset, calculate_test_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from helpers.classifier_helper import get_loaders, get_model
#from cellpose import models
import time
from collections import Counter
import argparse
# helpers/optimize_helper.py
from itertools import product

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_optimize_job(encoders, decoders, preprocessors, pretrains, in_folder, out_folder):
    """
    encoders, decoders, preprocessors, pretrains: iterable lists of choices (ints or strings)
    out_folder: path where results are written
    """
    read_new_blob_folder(in_folder)
    read_new_blob_folder(in_folder)
    combos = list(product(encoders, decoders, preprocessors, pretrains))
    total = len(combos)
    results = []
    print("[DEBUG] all combos:")
    for idx, (enc, dec, pre, pretrain) in enumerate(combos, start=1):
        print(enc, dec, pre, pretrain)
    for idx, (enc, dec, pre, pretrain) in enumerate(combos, start=1):
        print("[DEBUG] training:", enc, dec, pre, pretrain)
        res = train(encoder=enc, decoder=dec, preproc=pre, pretrain=pretrain, in_folder=in_folder, out_folder=out_folder)
        print(f"[DEBUG] training iteration done, result: {res}")
        results.append(res)
    best_result = np.max(results)
    return best_result

#def train(encoder, decoder, preproc, pretrain, out_folder):
#    print(encoder, decoder, preproc, pretrain)
#    return 0.5
def train(encoder, decoder, preproc, pretrain, in_folder, out_folder):
    pseudo_all_indices = []
    pseudo_all_labels = []

    with open(os.path.join(BASE_DIR, in_folder, 'pseudo_labels','pseudo_labels.json'), "r") as f:
        pseudo_label_json = json.load(f)
    for img_idx, blobs in pseudo_label_json.items():
        img_idx = int(img_idx)
        for blob_idx, lbl in blobs.items():
            pseudo_all_indices.append((img_idx, int(blob_idx)))
            pseudo_all_labels.append(int(lbl))

    with open(os.path.join(BASE_DIR, in_folder, 'image_blob_indices.pkl'), "rb") as f:
        all_indices = pickle.load(f)
    with open(os.path.join(BASE_DIR, in_folder, 'labels.pkl'), "rb") as f:
        all_labels = pickle.load(f)

    pseudo_train_idx, pseudo_val_idx, pseudo_train_lbls, pseudo_val_lbls = train_test_split(
        pseudo_all_indices,
        pseudo_all_labels,
        test_size=0.1,
        random_state=42,
        stratify=pseudo_all_labels
    )

    train_idx, val_idx, train_lbls, val_lbls = train_test_split(
        all_indices,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels  # optional: ensures label distribution is balanced
    )
    print("[DEBUG] print first few blob - label combos")
    for (img_idx, blob_idx), label in zip(train_idx[:5], train_lbls[:5]):
        print(f"Image {img_idx}, Blob {blob_idx} → Class {label}")



    #torch.cuda.empty_cache()
    #torch.cuda.ipc_collect()

    #model = models.CellposeModel(gpu=False)
    #enc = model.net.encoder

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] device: {device}")
    #enc.to(device)
    batch_size = 32


    blb = ClassificatorBlobHelper()
    pseudo_blb = ClassificatorBlobHelper(use_pseudo=True)
    print("[DEBUG] loaded new folder in Optimizer")

    train_transforms = Compose([
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),  # rotate in XY
        RandFlip(prob=0.5, spatial_axis=0),           # flip on Z
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),
        EnsureType()  # Makes sure it's a torch.Tensor
    ])

    train_dataset = ObjectPatchDataset(blb, train_idx, train_lbls, transform=train_transforms)
    val_dataset = ObjectPatchDataset(blb, val_idx, val_lbls)
    pseudo_train_dataset = ObjectPatchDataset(pseudo_blb, pseudo_train_idx, pseudo_train_lbls, transform=train_transforms)
    pseudo_val_dataset = ObjectPatchDataset(pseudo_blb, pseudo_val_idx, pseudo_val_lbls)


    model, reg_train_loader, reg_val_loader = get_model(encoder, decoder, preproc, pretrain, train_dataset, val_dataset)
    if pretrain == 'Semi-supervised':
        model, reg_train_loader, reg_val_loader = get_model(encoder, decoder, preproc, 'Kein Vortraining', train_dataset, val_dataset)
    print("[DEBUG] combo retreived, strat training")

    pseudo_train_loader, pseudo_val_loader = get_loaders(pseudo_train_dataset, pseudo_val_dataset)
    train_loader = reg_train_loader
    val_loader = reg_val_loader
    
    
    history = {"loss": [], "val_loss": [], "val_acc": []}

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    class_counts = Counter(train_lbls)
    print("Class distribution in training set:", class_counts)
    num_classes = len(class_counts)

    # Convert counts to weights (inverse frequency)
    class_weights = [len(train_lbls) / (num_classes * class_counts[c]) for c in sorted(class_counts.keys())]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_dir = "data/checkpoints"

    if pretrain == 'Semi-supervised':
        use_pseudo = True
    else:
        use_pseudo = False

    num_epochs = 80 if use_pseudo else 70

    checkpoint_path = os.path.join(checkpoint_dir, f"model_{encoder}{decoder}{pretrain}{preproc}.pt")
    history_path = os.path.join(checkpoint_dir, f"history_{encoder}{decoder}{pretrain}{preproc}.json")
    pseudo_checkpoint_path = os.path.join(checkpoint_dir, f"pseudo_model_{encoder}{decoder}{pretrain}{preproc}.pt")
    pseudo_history_path = os.path.join(checkpoint_dir, f"pseudo_history_{encoder}{decoder}{pretrain}{preproc}.json")
    # Initialize default history and state
    history = {"loss": [], "val_loss": [], "val_acc": []}
    start_epoch = 0
    best_val_acc = float('-inf')


    if os.path.exists(pseudo_checkpoint_path) and use_pseudo:
        print(f"Resuming training from checkpoint: {pseudo_checkpoint_path}")
        checkpoint = torch.load(pseudo_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint.get("val_acc", best_val_acc)
        print(f"Resuming training from epoch {start_epoch} for combo {encoder} {decoder} {pretrain} {preproc}")

        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            print(f"Resumed history with {len(history['loss'])} epochs")
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint.get("val_acc", best_val_acc)
        print(f"Resuming training from epoch {start_epoch} for version {encoder} {decoder} {pretrain} {preproc}")

        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            print(f"Resumed history with {len(history['loss'])} epochs")


    if use_pseudo:
        print("Using pseudo labels for training")
        train_loader = pseudo_train_loader
        val_loader = pseudo_val_loader
    else:
        print("Using regular labels for training")
        train_loader = reg_train_loader
        val_loader = reg_val_loader

    print(f"Number of learnable parameters: {len([p for p in model.parameters() if p.requires_grad])}")
    print("-"*100)
    print(f"Training version {encoder} {decoder} {pretrain} {preproc} with {len(train_dataset)} training samples and {len(val_dataset)} validation samples. Size of each picture: {train_dataset[0][0].shape}")
    print("-"*100)
    
    start_time = time.time()
    #for epoch in range(start_epoch, num_epochs):
    # ------------------------------------------------------------------
    # ZU TEST ZWECKEN VERKÜRTZT! TODO: RÜCKGÄNGIG
    # ------------------------------------------------------------------
    for epoch in range(2):
        if epoch > 40 and use_pseudo:
            print("Switching to regular labels for training")
            train_loader = reg_train_loader
            val_loader = reg_val_loader 
            use_pseudo = False
            model, _, _, _ = get_model(encoder, decoder, preproc, pretrain, train_dataset, val_dataset)
            if os.path.exists(pseudo_checkpoint_path) and use_pseudo:
                checkpoint = torch.load(pseudo_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
        

        model.train()
        running_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), (y - 1).to(device)

            # Debug shapes
            if batch_idx % 10 == 0:  # Print every 10 batches
                end_time = time.time()
                elapsed_time = end_time - start_time
                start_time = time.time()
                print(f"Epoch {epoch}, Batch {batch_idx} of {len(train_loader)}, time elapsed: {elapsed_time:.2f} seconds")

            # Forward
            #print(f"X shape: {X.shape}, y shape: {y.shape}") # X shape: torch.Size([32, 3, 32, 254, 254]), y shape: torch.Size([32])
            pred = model(X)
            loss = criterion(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        validation_time = time.time()
        
        val_loss, val_acc, predicted, labels = calculate_test_loss(model, val_loader, criterion, device) 
        validation_end_time = time.time()
        validation_elapsed_time = validation_end_time - validation_time
        avg_loss = running_loss / len(train_loader)
        print("-" * 50)
        print(f"[Epoch {epoch+1}/{num_epochs}]: Avg_loss = {avg_loss}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f} - Validation took {validation_elapsed_time:.2f} seconds")
        print("-" * 50)
        history["loss"].append(avg_loss)
        history["val_loss"].append(val_loss)    
        history["val_acc"].append(val_acc)    
        #print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # Save model checkpoint (overwrite to save space)
        if use_pseudo:
            checkpoint_path = os.path.join(checkpoint_dir, f"pseudo_model_{encoder}{decoder}{pretrain}{preproc}.pt")
            history_path = os.path.join(checkpoint_dir, f"pseudo_history_{encoder}{decoder}{pretrain}{preproc}.json")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_{encoder}{decoder}{pretrain}{preproc}.pt")
            history_path = os.path.join(checkpoint_dir, f"history_{encoder}{decoder}{pretrain}{preproc}.json")
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
            predicted = [int(x) for x in predicted]
            labels = [int(x) for x in labels]
            best_val_acc = val_acc  
            if use_pseudo:
                best_model_path = os.path.join(checkpoint_dir, f"best_pseudo_model_{encoder}{decoder}{pretrain}{preproc}.pt")
                val_acc_path = os.path.join(checkpoint_dir, f"val_acc_{encoder}{decoder}{pretrain}{preproc}.json")
                predictions_path = os.path.join(checkpoint_dir, f"predictions_{encoder}{decoder}{pretrain}{preproc}.json")
            else:
                best_model_path = os.path.join(checkpoint_dir, f"best_model_{encoder}{decoder}{pretrain}{preproc}.pt")
                val_acc_path = os.path.join(checkpoint_dir, f"pseudo_val_acc_{encoder}{decoder}{pretrain}{preproc}.json")
                predictions_path = os.path.join(checkpoint_dir, f"pseudo_predictions_{encoder}{decoder}{pretrain}{preproc}.json")

            val_acc_json = {"val_loss": val_loss, "val_acc": val_acc}
            
            with open(val_acc_path, "w") as f:
                json.dump(val_acc_json, f)
            with open(predictions_path, "w") as f:
                json.dump({"predicted": predicted, "labels": labels}, f)

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_acc": val_acc,
            }, best_model_path)
            
            print(f"New best model saved with val_acc = {val_acc:.4f} at: {best_model_path}")
    return best_val_acc
if __name__ == "__main__":
    train()
