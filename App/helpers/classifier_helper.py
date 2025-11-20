import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage as ndi
from torch.utils.data import DataLoader
from cellpose import models as cellpose_models
from skimage.measure import label as sklabel, regionprops, marching_cubes, mesh_surface_area, find_contours
from numpy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading
import json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load the masks
masks3D = []
masks3D.append(np.zeros((24,1024,1024)))
masks3D[0][11:13, 400:501, 400:501] = 1
#if os.path.exists(os.path.join(BASE_DIR,'data','masks.pkl')):
#    masks3D = []
#    with open(os.path.join(BASE_DIR,'data','masks.pkl'), 'rb') as f:
#        masks3D = pickle.load(f)

images3D_full = []
#with open(os.path.join(BASE_DIR,'data','images.pkl'), 'rb') as f:
#    images3D_full = pickle.load(f)
#images3D_full = [np.transpose(img, (0, 3, 1, 2)) for img in images3D_full]

images3D = []
images3D.append(np.zeros((24,1024,1024,4)))
images3D[0][11:13, 400:501, 400:501] = 1
#for im in images3D_full:
#    images3D.append(im[:,(0,1,3),:,:])




def normalize_channel(data):
    lower = np.percentile(data, 1)
    upper = np.percentile(data, 99)
    data = np.clip(data, lower, upper)
    return ((data - lower) / (upper - lower) * 255).astype(np.uint8)

def read_new_blob_folder(in_path):
    global images3D, masks3D
    print(f"reading from {in_path}")
    with open(os.path.join(BASE_DIR, 'data', in_path, 'masks.pkl'), 'rb') as f:
        masks3D = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'data', in_path, 'images.pkl'), 'rb') as f:
        images3D_full = pickle.load(f)
    images3D_full = [np.transpose(img, (0, 3, 1, 2)) for img in images3D_full]

    for im in images3D_full:
        images3D.append(im[:,(0,1,3),:,:])


class ClassificatorBlobHelper:
    def __init__(self):
        self.last_image = 0
        self.li = len(images3D)
        self.image = images3D[self.last_image]
        self.mask = masks3D[self.last_image]
        self.last_blob = 1
        self.lum = len(np.unique(self.mask))
        self.blob = self.mask == 1

    def reload_images(self, in_folder):
        print(f"[DEBUG] Reloading images from disk.")
        with open(os.path.join(BASE_DIR,'data', in_folder, 'masks.pkl'), 'rb') as f:
            masks3D = pickle.load(f)
            images3D_full = []
        with open(os.path.join(BASE_DIR,'data', in_folder, 'images.pkl'), 'rb') as f:
            images3D_full = pickle.load(f)
        images3D_full = [np.transpose(img, (0, 3, 1, 2)) for img in images3D_full]

        images3D = []
        for im in images3D_full:
            images3D.append(im[:,(0,1,3),:,:])

        self.last_image = 0
        self.image = images3D[self.last_image]
        self.mask = masks3D[self.last_image]

    def get_blob(self, image_index, blob_index, in_channel_dist=True, start_size = 64, offset=0, gaus_exp_nuc=10.0, gaus_exp_myo=20.0, binary_mask=False, z_size = 24):
        overflow = False
        if not self.last_image == image_index:
            # image index overflow protect
            if image_index >= self.li:
                image_index = image_index % self.li
                overflow = True
            # image index underflow protect
            if image_index < 0:
                image_index = self.li - image_index
            self.image = images3D[image_index]
            self.mask = masks3D[image_index]
            self.lum = len(np.unique(self.mask))
            self.last_image = image_index

        # blob index overflow protect
        if blob_index == self.lum:
            blob_index = 1
            self.image_index = image_index + 1
            if image_index >= self.li:
                image_index = image_index % self.li
                overflow = True
            print(f"[DEBUG] blob index overflow, set to 1")
            
        # blob index underflow protect
        if blob_index < 0:
            blob_index = self.lum + blob_index
            print(f"[DEBUG] blob index underflow, set to {blob_index}")

        if blob_index == 0:
            blob_index = self.lum - 1
            print(f"[DEBUG] blob index underflow, set to {blob_index}")


        #print(f"reading mask with {len(np.unique(self.mask))} blobs at {blob_index}")
        self.blob = self.mask == blob_index
        self.last_blob = blob_index
        
        if not np.any(self.blob):
            return None, None  

        # Get coordinates where mask is True
        coords = np.argwhere(self.blob)
        #z_min, z_max = 0, self.image.shape[0]
        z_min_abs = coords[:, 0].min()
        z_max_abs = coords[:, 0].max()
        x_min_abs = coords[:, 1].min()
        x_max_abs = coords[:, 1].max()
        y_min_abs = coords[:, 2].min()
        y_max_abs = coords[:, 2].max()

        x_center = (x_min_abs + x_max_abs) // 2
        y_center = (y_min_abs + y_max_abs) // 2
        z_center = (z_min_abs + z_max_abs) // 2
        # Desired size
        square_size = start_size + offset
        half_size = square_size // 2
        half_z = z_size // 2

        # Calculate initial x and y bounds
        x_min = x_center - half_size
        x_max = x_min + square_size
        y_min = y_center - half_size
        y_max = y_min + square_size
        z_min = z_center - half_z
        z_max = z_center + half_z

        # Correct for boundary clipping in x
        edge_blob = False
        if x_min < 0:
            x_max += -x_min
            x_min = 0
            x_min_abs += 2
        if x_max > self.mask.shape[1]:
            x_min -= x_max - self.mask.shape[1]
            x_max = self.mask.shape[1]
            x_max_abs -= 2
        x_min = max(x_min, 0)
        x_max = min(x_max, self.mask.shape[1])

        # Correct for boundary clipping in y
        if y_min < 0:
            y_max += -y_min
            y_min = 0
            y_min_abs += 2
        if y_max > self.mask.shape[2]:
            y_min -= y_max - self.mask.shape[2]
            y_max = self.mask.shape[2]
            y_max_abs -= 2
        y_min = max(y_min, 0)
        y_max = min(y_max, self.mask.shape[2])

        if z_min < 0:
            z_max += -z_min
            z_min = 0
            #z_min_abs += 2
        if z_max > self.mask.shape[0]:
            z_min -= z_max - self.mask.shape[0]
            z_max = self.mask.shape[0]
            #z_max_abs -= 2
        z_min = max(z_min, 0)
        z_max = min(z_max, self.mask.shape[0])

        #print(f"self.blob.shape: {self.blob.shape}") # shape: (D, H, W)
        cropped_mask = self.blob[z_min:z_max, x_min:x_max, y_min:y_max].transpose(1, 2, 0) # shape: (H, W, D)
        #print(f"[DEBUG] cropped_mask shape: {cropped_mask.shape}")
        if not binary_mask:
            # Compute distance transform from edges (inside mask)
            # Edge = mask - eroded_mask
            eroded = ndi.binary_erosion(cropped_mask)
            edges = cropped_mask ^ eroded
            distance = ndi.distance_transform_edt(~edges).astype(np.float32)  
            distance_nuc = np.exp(-distance / gaus_exp_nuc)
            #distance_myo = np.exp(-distance / gaus_exp_myo)

            cropped_img = self.image[z_min:z_max, :, x_min:x_max, y_min:y_max].transpose(1, 2, 3, 0).astype(np.float32)   # shape: (C, H, W, D)
            #print(f"[DEBUG] cropped_img shape: {cropped_img.shape}")
            distance_nuc = (distance_nuc / distance_nuc.max()).astype(np.float32)  # Normalize to [0, 1]
            #distance_myo = (distance_myo / distance_myo.max()).astype(np.float32)  # Normalize to [0, 1]

            cropped_img = cropped_img.astype(np.float32)

            if in_channel_dist:
                #coords = np.argwhere(cropped_mask)  # shape: (N, 3), each row is [H, W, D]
                #z_min_crop = coords[:, 2].min()
                #z_max_crop = coords[:, 2].max()
                #cropped_img[:, :, :, :z_min_crop] = 0
                #cropped_img[:, :, :, z_max_crop+1:] = 0

                outside_mask = cropped_mask == 0

                cropped_img[0][outside_mask] *= distance_nuc[outside_mask]
                #cropped_img[1][outside_mask] *= distance_myo[outside_mask]

                #final_blob = np.concatenate([cropped_img, cropped_mask[np.newaxis] * 255], axis=0).astype(np.uint8)
                final_blob = cropped_img.astype(np.uint8)
                #print(f"Final blob shape: {final_blob.shape} with in_channel_dist={in_channel_dist}")
            else:
                # Add distance as an extra channel (version 2)
                final_blob = np.concatenate([
                    cropped_mask[np.newaxis] * 255,
                    cropped_img[1][np.newaxis],
                    cropped_img[2][np.newaxis]
                ], axis=0).astype(np.uint8)
                #final_blob = np.concatenate([cropped_mask[np.newaxis] * 255, cropped_img[1], cropped_img[2]], axis=0).astype(np.uint8)# distance_nuc[np.newaxis] * 255, cropped_mask[np.newaxis] * 255], axis=0).astype(np.uint8)  # shape: (4, H, W, D)
                #print(f"Final blob shape: {final_blob.shape} with in_channel_dist={in_channel_dist}")
        else:         # Extract only bounding box in spatial dims, full depth
            #print(f"[DEBUG] cutting binary mask")
            cropped_img = add_padding(self.image[z_min:z_max, :, x_min_abs:x_max_abs+1, y_min_abs:y_max_abs+1].transpose(1, 2, 3, 0).astype(np.float32), start_size, start_size)  # (C, H, W, D)
            #print(f"[DEBUG] cropped_img shape: {cropped_img.shape}")
            cropped_mask = add_padding(self.blob[z_min:z_max, x_min_abs:x_max_abs+1, y_min_abs:y_max_abs+1].transpose(1, 2, 0), start_size, start_size)  # (H, W, D)
            #print(f"[DEBUG] cropped_mask shape: {cropped_mask.shape}")

            #coords = np.argwhere(self.blob)
            #z_min = coords[:, 0].min()
            #z_max = coords[:, 0].max()
            bbox_mask = np.zeros_like(cropped_mask, dtype=np.float32)
            bbox_mask[z_min:z_max, x_min_abs:x_max_abs+1, y_min_abs:y_max_abs+1] = 1.0  # Set the bounding box area to 1.0

            # Zero out pixels outside mask
            masked_nuc = cropped_img[0] * bbox_mask
            masked_myo = cropped_img[1]

            # Stack into 3 channels: nuc, myo, mask (all float32)
            final_blob = np.stack([masked_myo.astype(np.float32), masked_nuc.astype(np.float32), cropped_mask.astype(np.float32)], axis=0)
        return final_blob, overflow
    
import torch.nn.functional as F

def bicubic_upsample_3d(img_tensor, target_size=(256, 256)):
    """
    Upsample a 4D tensor (C, H, W, D) spatially using bicubic interpolation.
    Assumes input is a NumPy array or PyTorch tensor in (C, H, W, D).
    Upsampling is done per-slice (along D axis).

    Returns:
        torch.Tensor of shape (C, target_H, target_W, D)
    """
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor)

    #print(f"[DEBUG] blob shape: {img_tensor.shape}")
    C, H, W, D = img_tensor.shape
    upsampled_slices = []

    for i in range(D):
        slice_i = img_tensor[..., i].unsqueeze(0)  # (C, H, W)
        slice_i_up = F.interpolate(slice_i, size=target_size, mode='bicubic', align_corners=False)
        upsampled_slices.append(slice_i_up)

    upsampled = torch.stack(upsampled_slices, dim=-1).squeeze(0)  # (C, H_up, W_up, D)
    return upsampled


def add_padding(blob, target_H, target_W):
    if blob.ndim == 4:
        H, W = blob.shape[1], blob.shape[2]
    # For 3D array: (H, W, D)
    elif blob.ndim == 3:
        H, W = blob.shape[0], blob.shape[1]
    pad_H = max(target_H - H, 0)
    pad_W = max(target_W - W, 0)

    # Calculate padding before and after for height and width
    pad_top = pad_H // 2
    pad_bottom = pad_H - pad_top
    pad_left = pad_W // 2
    pad_right = pad_W - pad_left

    # For 4D array: (C, H, W, D)
    if blob.ndim == 4:
        padding = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    # For 3D array: (H, W, D)
    elif blob.ndim == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        raise ValueError(f"Unexpected array shape {blob.shape}")

    return np.pad(blob, padding, mode='constant', constant_values=0)


def preprocess_blob(blob, min_depth=24):
    #print(f"[DEBUG] IN Blob Max: {blob.max()}, Min: {blob.min()}")
    # blob: torch.Tensor of shape (3, H, W, D)
    assert isinstance(blob, torch.Tensor)
    #assert blob.shape[0] == 3

    # Normalize intensity channels (example: z-score)
    for i in range(2):  # channels 0 and 1 are image channels
        c = blob[i]
        mean = c.mean()
        std = c.std()
        blob[i] = (c - mean) / (std + 1e-5)
        
        c_min = c.min()
        c_max = c.max()
        blob[i] = (c - c_min) / (c_max - c_min + 1e-5)  # now in [0, 1]
        blob[i] = blob[i] * 255    

    # Rearrange to (C, D, H, W)
    blob = blob.permute(0, 3, 1, 2)  # (3, H, W, D) -> (3, D, H, W)

    # Pad depth (D) to min_depth if needed
    depth = blob.shape[1]
    if depth < min_depth:
        pad_amount = min_depth - depth
        # Pad at the end of depth axis (dimension 1)
        blob = F.pad(blob, (0, 0, 0, 0, 0, pad_amount))  # (W, H, D)
        #print(f"Padded blob depth from {depth} to {blob.shape[1]}")

    #print(f"blob shape out: {blob.shape}")  # should be (3, D, H, W)

    #print(f"[DEBUG] OUT Blob Max: {blob.max()}, Min: {blob.min()}")
    return blob


from torch.utils.data import Dataset

class ObjectPatchDataset(Dataset):
    """
    blob_helper: an instance of ClassificatorBlobHelper
    image_blob_indices: list of (image_index, blob_index) tuples
    labels: list of int class labels corresponding to each blob
    version: 1 - use in_channel_dist=True, 2 - use in_channel_dist=False
    """
    def __init__(self, blob_helper, image_blob_indices, labels, transform=None):
        self.blob_helper = blob_helper
        self.indices = image_blob_indices
        self.labels = labels
        self.transform = transform
        self.version = 1
        #print(f"[DEBUG] ObjectPatchDataset initialized with {len(self.indices)} indices and {len(self.labels)} labels.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx, blob_idx = self.indices[idx]
        #print(f"[DEBUG] Fetching item {idx}: Image {img_idx}, Blob {blob_idx}")
        label = self.labels[idx]
        if self.version == 1:
            #print("-"*100)
            #print(f"[DEBUG] Using version 1 with in_channel_dist=True", flush=True)
            #print("-"*100)
            blob, oob = self.blob_helper.get_blob(img_idx, blob_idx, in_channel_dist=True)
        elif self.version == 2:
            #print("-"*100)
            #print(f"[DEBUG] Using version 2 with in_channel_dist=False", flush=True)
            #print("-"*100)
            #print(f"[DEBUG] getting blob for image {img_idx}, blob {blob_idx}, label {label}")
            blob, oob = self.blob_helper.get_blob(img_idx, blob_idx, in_channel_dist=False, binary_mask=False)
            if blob is None: raise IndexError(f"Empty blob for image {img_idx} blob {blob_idx}")
            #else: print(f"[DEBUG] Retrieved blob shape: {blob.shape} for image {img_idx}, blob {blob_idx}, label {label}, oob={oob}")
            blob = bicubic_upsample_3d(blob, (256, 256))  # Ensure blob is padded to 256
        elif self.version == 3:
            #print("-"*100)
            #print(f"[DEBUG] Using version 3 with binary_mask=True", flush=True)
            #print("-"*100)
            blob, oob = self.blob_helper.get_blob(img_idx, blob_idx, in_channel_dist=True, binary_mask=True)
        elif self.version == 4:
            #print("-"*100)
            #print(f"[DEBUG] Using version 3 with binary_mask=True", flush=True)
            #print("-"*100)
            blob, oob = self.blob_helper.get_blob(img_idx, blob_idx, in_channel_dist=True, binary_mask=False)
            blob = add_padding(blob, 256, 256)  # Ensure blob is padded to 256
        elif self.version == 5:
            #print("-"*100)
            #print(f"[DEBUG] Using version 3 with binary_mask=True", flush=True)
            #print("-"*100)
            blob, oob = self.blob_helper.get_blob(img_idx, blob_idx, in_channel_dist=True, binary_mask=False)
            blob = bicubic_upsample_3d(blob, (256, 256))  # Ensure blob is padded to 256

        
        if self.transform:
            blob = self.transform(blob)
            #print(f"[DEBUG] Transforming complete")
        else:
            if isinstance(blob, np.ndarray):
                blob = torch.from_numpy(blob)

        
        blob = blob.float()

        blob = preprocess_blob(blob)
        #print(f"[DEBUG] Blob Max: {blob.max()}, Min: {blob.min()}, Shape: {blob.shape}, Version: {self.version}")
        return blob, label
    def change_version(self, version):
        """
        Change the version of the dataset to change the behavior of how the segmentation mask plays into the fixed size image
        version: 1 - use in_channel_dist=True resulting in 3 channels, each scaled by distance transform, 
        version: 2 - use in_channel_dist=False resulting in 4 channels, the second one being distance transform,
        version: 3 - use in_channel_dist=False resulting in 3 channels, the first two being a cutout of the minimal bounding box of the blob and the last one being the mask.
        """
        #print("-"*100)
        print(f"[DEBUG] changing to version {version}")
        #print("-"*100)
        self.version = version

def calculate_test_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), (labels - 1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy, all_predicted, all_labels


#X shape: torch.Size([32, 3, 32, 254, 254]), y shape: torch.Size([32])
class LearnableUpsampler(nn.Module):
    def __init__(self, input_size=64, output_size=256):
        super().__init__()
        scale_factor = output_size // input_size
        self.upsampler = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.upsampler(x)


class SAMClassifierMLP(nn.Module):
    def __init__(self, pretrained_ViT, num_classes=5, learnable_upsample=False, D=24):
        super().__init__()
        self.encoder = pretrained_ViT
        self.learnable_upsample = learnable_upsample
        if learnable_upsample:
            self.upsampler = LearnableUpsampler()

        self.classifier = nn.Sequential(
            nn.Linear(D * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


        self.depth_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # [B*D, C, H, W]

        if self.learnable_upsample:
            x = self.upsampler(x)  # [B*D, C, 256, 256]

        feats = self.encoder(x)  # [B*D, 256, 32, 32]
        feats = feats.view(B, D, 256, -1).mean(dim=-1)  # [B, D, 256]
        attn_output, _ = self.depth_attention(feats, feats, feats)  # [B, D, 256]

        # New part: Flatten and pass to classifier
        flattened_feats = attn_output.reshape(B, -1)  # [B, D * 256]
        #print(f"[DEBUG] Flattened feats shape: {flattened_feats.shape}")  # Debugging line
        out = self.classifier(flattened_feats)
        return out




class SAMClassifier3D(nn.Module):
    def __init__(self, pretrained_ViT, num_classes=5, learnable_upsample=False):
        super().__init__()
        self.encoder = pretrained_ViT
        self.learnable_upsample = learnable_upsample
        if learnable_upsample:
            self.upsampler = LearnableUpsampler()

        self.classifier3d = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # [B*D, C, H, W]

        if self.learnable_upsample:
            x = self.upsampler(x)  # [B*D, C, 256, 256]

        feats = self.encoder(x)  # [B*D, 256, H_out, W_out]
        _, C_feat, H_out, W_out = feats.shape
        feats = feats.view(B, D, C_feat, H_out, W_out).permute(0, 2, 1, 3, 4)

        out = self.classifier3d(feats)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMClassifier3D_CENTER_AWARE(nn.Module):
    def __init__(self, pretrained_ViT, num_classes=5, learnable_upsample=False, center_crop=False):
        super().__init__()
        self.encoder = pretrained_ViT
        self.feats = None
        self.out_feats = None  
        self.cropped_feats = None
        self.center_crop = center_crop 
        self.learnable_upsample = learnable_upsample
        if learnable_upsample:
            self.upsampler = LearnableUpsampler()

        self.classifier3d = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.linear = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 512),
            nn.Linear(512, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        #print(f"[DEBUG] Input shape: {x.shape}")  # Debugging line
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # [B*D, C, H, W]

        if self.learnable_upsample:
            x = self.upsampler(x)  # [B*D, C, 256, 256]

        feats = self.encoder(x)  # [B*D, 256, H_out, W_out]
        _, C_feat, H_out, W_out = feats.shape
        feats = feats.view(B, D, C_feat, H_out, W_out).permute(0, 2, 1, 3, 4)  # [B, C_feat, D, H, W]
        self.feats = feats  # Store for later use
        x = self.classifier3d(feats)  # [B, 128, D, H, W]
        self.out_feats = x  
        # --- Center Crop ---
        if self.center_crop:
            d, h, w = x.shape[-3:]
            crop_size = int(min(d, h, w) * 0.8)  # take 80% of each dim
            start_d = (d - crop_size) // 2
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            x = x[:, :, start_d:start_d+crop_size, start_h:start_h+crop_size, start_w:start_w+crop_size]
            self.cropped_feats = x  

        # --- Adaptive Average Pooling to (4x4x4) ---
        x = F.adaptive_avg_pool3d(x, (4, 4, 4))  # [B, 128, 4, 4, 4]
        x = x.view(x.size(0), -1)                # Flatten to [B, 128*4*4*4]
        out = self.linear(x)                     # Final classification layer
        return out

import torchvision.models as models
def get_resnet18_encoder(pretrained=True, semiPretrained=False):
    if pretrained:
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        encoder = nn.Sequential(*list(resnet.children())[:-2])
        #for param in encoder.parameters():
        #    param.requires_grad = False
        stopCrit = int(len(list(resnet.parameters())) * 0.7)
        for i, param in enumerate(resnet.parameters()):
            if i < stopCrit:
                param.requires_grad = False
    else:
        resnet = models.resnet18(weights=None)
        encoder = nn.Sequential(*list(resnet.children())[:-2])
        if semiPretrained:
            stopCrit = int(len(list(resnet.parameters())) * 0.7)
            for i, param in enumerate(encoder.parameters()):
                if i < stopCrit:
                    param.requires_grad = False
        
    encoder = nn.Sequential(
        encoder,
        nn.Conv2d(512, 256, kernel_size=1)
    )
    print(f"Number of learnable parameters: {len([p for p in encoder.parameters() if p.requires_grad])}")
    return encoder


def get_resnet101_encoder(pretrained=True, semiPretrained=False):
    if pretrained:
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        encoder = nn.Sequential(*list(resnet.children())[:-2])
        #for param in encoder.parameters():
        #    param.requires_grad = False
        stopCrit = int(len(list(resnet.parameters())) * 0.7)
        for i, param in enumerate(resnet.parameters()):
            if i < stopCrit:
                param.requires_grad = False
    else:
        resnet = models.resnet101(weights=None)
        encoder = nn.Sequential(*list(resnet.children())[:-2])
        if semiPretrained:
            stopCrit = int(len(list(resnet.parameters())) * 0.7)
            for i, param in enumerate(encoder.parameters()):
                if i < stopCrit:
                    param.requires_grad = False
    
    encoder = nn.Sequential(
        encoder,
        nn.Conv2d(2048, 256, kernel_size=1)  # adapt channels
    )
    print(f"Number of learnable parameters: {len([p for p in encoder.parameters() if p.requires_grad])}")
    return encoder

def get_efficientnetv2l_encoder(pretrained=True, semiPretrained=False):
    if pretrained:
        eff = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        #for param in eff.features.parameters():
        #    param.requires_grad = False
        stopCrit = int(len(list(eff.parameters())) * 0.7)
        for i, param in enumerate(eff.parameters()):
            if i < stopCrit:
                param.requires_grad = False
    else:
        eff = models.efficientnet_v2_l(weights=None)
        if semiPretrained:
            stopCrit = int(len(list(eff.parameters())) * 0.7)
            for i, param in enumerate(eff.parameters()):
                if i < stopCrit:
                    param.requires_grad = False

    encoder = nn.Sequential(*list(eff.features), nn.Conv2d(1280, 256, kernel_size=1))
    print(f"Number of learnable parameters: {len([p for p in encoder.parameters() if p.requires_grad])}")
    return encoder

def get_convnextxl_encoder(pretrained=True, semiPretrained=False):
    if pretrained:
        convn = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        #for param in convn.features.parameters():
        #    param.requires_grad = False
        stopCrit = int(len(list(convn.parameters())) * 0.7)
        for i, param in enumerate(convn.parameters()):
            if i < stopCrit:
                param.requires_grad = False
    else:
        convn = models.convnext_large(weights=None)
        if semiPretrained:
            stopCrit = int(len(list(convn.parameters())) * 0.7)
            for i, param in enumerate(convn.parameters()):
                if i < stopCrit:
                    param.requires_grad = False
    
    encoder = nn.Sequential(
        *list(convn.features),
        nn.Conv2d(1536, 256, kernel_size=1)  
    )
    print(f"Number of learnable parameters: {len([p for p in encoder.parameters() if p.requires_grad])}")
    return encoder

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        #print(f"[DEBUG] Permute input shape: {x.shape}")
        x = x.permute(*self.dims)
        #print(f"[DEBUG] Permute output shape: {x.shape}")
        return x

def get_swinl_encoder(pretrained=True, semiPretrained=False):
    if pretrained:
        swin = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
        #for param in swin.features.parameters():
        #    param.requires_grad = False
        stopCrit = int(len(list(swin.parameters())) * 0.7)
        for i, param in enumerate(swin.parameters()):
            if i < stopCrit:
                param.requires_grad = False
    else:
        swin = models.swin_v2_b(weights=None)
        if semiPretrained:
            stopCrit = int(len(list(swin.parameters())) * 0.7)
            for i, param in enumerate(swin.parameters()):
                if i < stopCrit:
                    param.requires_grad = False    

    encoder = nn.Sequential(
        swin.features,
        Permute(0, 3, 1, 2),     # (B,H,W,C) → (B,C,H,W)
        nn.Conv2d(1024, 256, 1)
    )
    print(f"Number of learnable parameters: {len([p for p in encoder.parameters() if p.requires_grad])}")
    return encoder

def get_SAMViT_encoder(pretrained, semiPretrained=True):
    model_cpsam = cellpose_models.CellposeModel(gpu=False)
    enc = model_cpsam.net.encoder
    if semiPretrained:
        stopCrit = int(len(list(enc.parameters())) * 0.7)
        for i, param in enumerate(enc.parameters()):
            if i < stopCrit:
                param.requires_grad = False    
    #for param in enc.parameters():
    #    param.requires_grad = False
    return enc

def get_densenet_vanilla():
    return DenseNet121(spatial_dims=3, in_channels=3, out_channels=4)

def get_densenet264(in_ch=4, n_classes=4):
    return DenseNet264(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=n_classes,
        init_features=96,  
        growth_rate=48,    
        pretrained=False,
    )
def get_densenetMAX(in_ch=4, n_classes=4):
    return DenseNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=n_classes,
        init_features=128,
        growth_rate=64,
        block_config=(8, 16, 80, 64),  
        bn_size=4,
        dropout_prob=0.0,
    )

def get_senet154(in_ch=4, n_classes=4):
    return SENet154(
        spatial_dims=2,
        in_channels=in_ch,
        num_classes=n_classes,
        pretrained=False,
    )

def get_seresnext101_heavy(in_ch=4, n_classes=4):
    return SEResNext101(
        spatial_dims=2,
        in_channels=in_ch,
        num_classes=n_classes,
        pretrained=False,
    )

def get_efficientnet_l2(in_ch=4, n_classes=4):
    return EfficientNetBN(
        model_name="efficientnet-l2",
        spatial_dims=2,
        in_channels=in_ch,
        num_classes=n_classes,
        pretrained=False,  
    )

from monai.networks.nets import (
    DenseNet264, DenseNet,
    SENet154, SEResNext101,
    EfficientNetBN,
    ViT, DenseNet121
)

def get_model(encoder, decoder, preproc, pretrain, train_dataset, val_dataset, inference=False):
    print(encoder, decoder, preproc, pretrain)
    if preproc == 'Segmentierungsmaske':
        preproc = 5
    elif preproc == 'Distanztransformation':
        preproc = 2
    else:
        print(f"[ERROR] no Preprocessing \"{preproc}\" found, use Segmentierungsmaske")
    if not inference:
        train_dataset.change_version(preproc)
        val_dataset.change_version(preproc)
    
    if pretrain == 'Kein Vortraining':
        pretrained = False
        semi_pretrained = False
    elif pretrain == 'Semi-supervised':
        pretrained = False
        semi_pretrained = True
    elif pretrain == 'Fully-supervised':
        pretrained = True
        semi_pretrained = False
    else:
        print(f"[ERROR] no pretrain \"{pretrain}\" found, use Fully-supervised")
        pretrained = True
        semi_pretrained = False

    if encoder == 'CellposeSAM':
        enc = get_SAMViT_encoder(pretrained=pretrained, semiPretrained=semi_pretrained)
    elif encoder == 'ResNet18':
        enc = get_resnet18_encoder(pretrained=pretrained, semiPretrained=semi_pretrained)
    elif encoder == 'ResNet101':
        enc = get_resnet101_encoder(pretrained=pretrained, semiPretrained=semi_pretrained)
    elif encoder == 'SwinV2':
        enc = get_swinl_encoder(pretrained=pretrained, semiPretrained=semi_pretrained)
    elif encoder == 'ConvNeXt':
        enc = get_convnextxl_encoder(pretrained=pretrained, semiPretrained=semi_pretrained)
    elif encoder == 'EfficientNetV2':
        enc = get_efficientnetv2l_encoder(pretrained=pretrained, semiPretrained=semi_pretrained)
    else:
        print(f"[ERROR] no encoder \"{encoder}\" found, use ResNet18")
        enc = get_resnet18_encoder(pretrained=pretrained, semiPretrained=semi_pretrained)

    if decoder == 'Schichten-Klassifikator':
        model = SAMClassifierMLP(enc, num_classes=4)
    elif decoder == 'Volumen-Klassifikator':
        model = SAMClassifier3D_CENTER_AWARE(enc, num_classes=4)
    else:
        print(f"[ERROR] no decoder \"{decoder}\" found, use Volumen-Klassifikator")
        model = SAMClassifier3D_CENTER_AWARE(enc, num_classes=4)
        
    if not inference:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)
        print(f"successfully loaded model and Dataloader of size {len(train_loader.dataset)} with {len(train_loader)} batches")
    else:
        train_loader = None
        val_loader = None
               
    return model, train_loader, val_loader


def get_loaders(train_dataset, val_dataset, batch_size=1):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def extract_features_3d(mask3D):
    labels = np.unique(mask3D)
    labels = labels[labels != 0]
    out = {}
    for label_id in labels:
        if label_id % 100 == 0:
            print(f"Feature Extractor processing label {label_id} / {labels[-1]}")
        binmask = (mask3D == label_id)
        if not np.any(binmask):
            continue
        vol = int(binmask.sum())
        z_any = binmask.any(axis=(1,2))
        num_slices = int(z_any.sum())
        labeled = sklabel(binmask.astype(np.uint8))
        props = regionprops(labeled)
        if len(props) == 0:
            continue
        p = props[0]
        bbox = p.bbox
        if len(bbox) == 6:
            dz = bbox[3] - bbox[0]
            dy = bbox[4] - bbox[1]
            dx = bbox[5] - bbox[2]
        else:
            dz = 1
            dy = bbox[2] - bbox[0]
            dx = bbox[3] - bbox[1]
        bbox_vol = max(int(dz * dy * dx), 1)
        extent = float(vol) / float(bbox_vol)
        try:
            verts, faces, _, _ = marching_cubes(binmask.astype(np.uint8), level=0.5)
            surf_area = float(mesh_surface_area(verts, faces))
        except Exception:
            surf_area = 0.0
        try:
            maj = float(p.major_axis_length)
        except Exception:
            maj = 0.0
        try:
            minr = float(p.minor_axis_length)
        except Exception:
            minr = 0.0
        try:
            ecc = float(p.eccentricity)
        except Exception:
            ecc = 0.0
        slice_indices = np.where(z_any)[0]
        if slice_indices.size > 0:
            areas = [int(binmask[z].sum()) for z in slice_indices]
            z_max = int(slice_indices[int(np.argmax(areas))])
        else:
            z_max = 0
        slice_mask = binmask[z_max].astype(np.uint8)
        contours = find_contours(slice_mask, 0.5)
        if len(contours) == 0:
            fourier_abs = np.zeros(10, dtype=float)
            best_contour = np.zeros((0,2), dtype=float)
        else:
            lengths = [c.shape[0] for c in contours]
            best_contour = contours[int(np.argmax(lengths))]
            complex_contour = best_contour[:,1] + 1j * best_contour[:,0]
            f = fft(complex_contour)
            fourier_abs = np.abs(f[:10])
            if fourier_abs.shape[0] < 10:
                fourier_abs = np.pad(fourier_abs, (0, 10 - fourier_abs.shape[0]))
        features = [
            vol,
            surf_area,
            extent,
            num_slices,
            maj,
            minr,
            ecc,
            *fourier_abs.tolist()
        ]
        out[int(label_id)] = {
            "features": features,
            "max_area_slice": z_max,
            "contour": best_contour
        }
    return out

def cluster_features_3d(masks3D_list, label_stores_list):
    pseudo_label_json = {}
    for img_idx, mask3D in enumerate(masks3D_list):
        print("*" * 50)
        print(f"Clustering features for image {img_idx} / {len(masks3D_list)}")
        label_store = label_stores_list[img_idx]
        features_dict = extract_features_3d(mask3D)
        print(f"Extracted features for {len(features_dict)} instances")

        labels = []
        X = []
        for k, v in features_dict.items():
            X.append(v["features"])
            labels.append(k)
        X = np.array(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Features scaled")

        pca = PCA(n_components=10, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        print("PCA completed")

        y = np.full(len(labels), -1)
        for idx, inst_id in enumerate(labels):
            if str(inst_id) in label_store:
                y[idx] = label_store[str(inst_id)]

        label_spread = LabelSpreading(kernel="rbf", alpha=0.2)
        y_spread = label_spread.fit(X_reduced, y).transduction_

        pseudo_labels = {str(inst_id): int(lbl) for inst_id, lbl in zip(labels, y_spread)}
        pseudo_label_json[str(img_idx)] = pseudo_labels
        
        
        print("*" * 50)

    return pseudo_label_json


def create_pseudo_labels(in_folder, label_stores_list):
    masks = []
    with open(os.path.join(BASE_DIR,'data', in_folder, 'masks.pkl'), 'rb') as f:
        masks = pickle.load(f)
    pseudo_label_json = cluster_features_3d(masks, label_stores_list)

    with open(os.path.join(BASE_DIR,'data', in_folder, "pseudo_labels.json"), "w") as f:
        json.dump(pseudo_label_json, f, indent=2)

    return pseudo_label_json