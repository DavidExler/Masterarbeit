import pickle
import numpy as np
import torch
import scipy.ndimage as ndi
#from helpers.visualization_helper import normalize_with_cutoffs
# load the masks
masks3D_20xRenamed = []
#with open('helpers/masks3D_CELLPOSE_RUN_1.pkl', 'rb') as f:
#    masks3D_20xRenamed = pickle.load(f)
with open('save_data/masks/20xRenamed/masks3D_CELLPOSE_RUN_1.pkl', 'rb') as f:
    masks3D_20xRenamed = pickle.load(f)
blobs_per_image = [len(np.unique(mask)) for mask in masks3D_20xRenamed]

images3D_20xRenamed_full = []
with open('save_data/imgs_20xRenamed.pkl', 'rb') as f:
    images3D_20xRenamed_full = pickle.load(f)

images3D_20xRenamed = []
for im in images3D_20xRenamed_full:
    images3D_20xRenamed.append(im[:,0:2,:,:])
#li = len(images3D_20xRenamed)


def normalize_channel(data):
    lower = np.percentile(data, 1)
    upper = np.percentile(data, 99)
    data = np.clip(data, lower, upper)
    return ((data - lower) / (upper - lower) * 255).astype(np.uint8)


class ClassificatorBlobHelper:
    def __init__(self):
        self.li = len(images3D_20xRenamed)
        self.last_image = 0
        self.last_blob = 1
        self.image = images3D_20xRenamed[self.last_image]
        self.mask = masks3D_20xRenamed[self.last_image]
        self.lum = len(np.unique(self.mask))
        self.blob = self.mask == 1

    def get_blob(self, image_index, blob_index, in_channel_dist=True, start_size = 64, offset=0, gaus_exp_nuc=10.0, gaus_exp_myo=20.0, binary_mask=False):
        # Load new image only if necessary
        if not self.last_image == image_index:
            # image index overflow protect
            image_index = image_index % self.li
            # image index underflow protect
            if image_index < 0:
                image_index = self.li - image_index

            self.image = images3D_20xRenamed[image_index]
            self.mask = masks3D_20xRenamed[image_index]
            self.lum = len(np.unique(self.mask))
            print(f"image changed from {self.last_image} to {image_index} -> loaded img of shape {self.image.shape}  and mask of shape {self.mask.shape} which has {self.lum} indices. The selected blob is {blob_index}")
        
        #else:
        #    print(f"still at {self.last_image} == {image_index} -> Still img of shape {self.image.shape} and mask of shape {self.mask.shape} which has {self.lum} indices. The selected blob is {blob_index}")
        self.last_image = image_index

        mistake = False
        # blob index overflow protect
        if blob_index == self.lum:
            blob_index = 1
            print(f"[DEBUG] blob index overflow, set to 1")
            mistake = True
        # blob index underflow protect
        if blob_index < 0:
            blob_index = self.lum + blob_index
            print(f"[DEBUG] blob index underflow, set to {blob_index}")
            mistake = True
        if blob_index == 0:
            blob_index = self.lum - 1
            print(f"[DEBUG] blob index underflow, set to {blob_index}")
            mistake = True
        self.blob = self.mask == blob_index
        self.last_blob = blob_index
        
        if not np.any(self.blob):
            return None, None  

        # Get coordinates where mask is True
        coords = np.argwhere(self.blob)
        z_min, z_max = 0, self.image.shape[0]
        x_min_abs = coords[:, 1].min()
        x_max_abs = coords[:, 1].max()
        y_min_abs = coords[:, 2].min()
        y_max_abs = coords[:, 2].max()

        x_center = (x_min_abs + x_max_abs) // 2
        y_center = (y_min_abs + y_max_abs) // 2

        # Desired size
        square_size = start_size + offset
        half_size = square_size // 2

        # Calculate initial x and y bounds
        x_min = x_center - half_size
        x_max = x_min + square_size
        y_min = y_center - half_size
        y_max = y_min + square_size

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

        #print(f"self.blob.shape: {self.blob.shape}") # shape: (D, H, W)
        cropped_mask = self.blob[z_min:z_max, x_min:x_max, y_min:y_max].transpose(1, 2, 0) # shape: (H, W, D)

        if not binary_mask:
            # Compute distance transform from edges (inside mask)
            # Edge = mask - eroded_mask
            eroded = ndi.binary_erosion(cropped_mask)
            edges = cropped_mask ^ eroded
            distance = ndi.distance_transform_edt(~edges).astype(np.float32)  
            distance_nuc = np.exp(-distance / gaus_exp_nuc)
            distance_myo = np.exp(-distance / gaus_exp_myo)

            cropped_img = self.image[z_min:z_max, :, x_min:x_max, y_min:y_max].transpose(1, 2, 3, 0).astype(np.float32)   # shape: (C, H, W, D)
            distance_nuc = (distance_nuc / distance_nuc.max()).astype(np.float32)  # Normalize to [0, 1]
            distance_myo = (distance_myo / distance_myo.max()).astype(np.float32)  # Normalize to [0, 1]

            cropped_img = cropped_img.astype(np.float32)

            if in_channel_dist:
                cropped_img[0] *= distance_nuc
                cropped_img[1] *= distance_myo
                final_blob = np.concatenate([cropped_img, cropped_mask[np.newaxis] * 255], axis=0).astype(np.uint8)
                #print(f"Final blob shape: {final_blob.shape} with in_channel_dist={in_channel_dist}")
            else:
                # Add distance as an extra channel (version 2)
                final_blob = np.concatenate([cropped_img, distance_nuc[np.newaxis] * 255, cropped_mask[np.newaxis] * 255], axis=0).astype(np.uint8)  # shape: (4, H, W, D)
                #print(f"Final blob shape: {final_blob.shape} with in_channel_dist={in_channel_dist}")
        else:         # Extract only bounding box in spatial dims, full depth
            cropped_img = self.image[z_min:z_max, :, x_min_abs:x_max_abs+1, y_min_abs:y_max_abs+1].transpose(1, 2, 3, 0).astype(np.float32)  # (C, H, W, D)
            cropped_mask = self.blob[z_min:z_max, x_min_abs:x_max_abs+1, y_min_abs:y_max_abs+1].transpose(1, 2, 0)  # (H, W, D)

            # Zero out pixels outside mask
            masked_nuc = cropped_img[0] * cropped_mask  # (H, W, D)
            masked_myo = cropped_img[1] * cropped_mask  # (H, W, D)

            # Stack into 3 channels: nuc, myo, mask (all float32)
            final_blob = np.stack([masked_myo.astype(np.float32), masked_nuc.astype(np.float32), cropped_mask.astype(np.float32)], axis=0)
        return final_blob, mistake
    
import torch.nn.functional as F

def preprocess_blob(blob, min_depth=32):
    #print(f"blob shape in: {blob.shape}")
    # blob: torch.Tensor of shape (3, H, W, D)
    assert isinstance(blob, torch.Tensor)
    #assert blob.shape[0] == 3

    # Normalize intensity channels (example: z-score)
    for i in range(2):  # channels 0 and 1 are image channels
        c = blob[i]
        mean = c.mean()
        std = c.std()
        blob[i] = (c - mean) / (std + 1e-5)

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

    def __len__(self):
        return len(self.indices)

    #def __getitem__(self, idx):
    #    img_idx, blob_idx = self.indices[idx]
    #    label = self.labels[idx]
    #    blob, oob = self.blob_helper.get_blob(img_idx, blob_idx)
    #
    #    if isinstance(blob, np.ndarray):
    #        blob = torch.from_numpy(blob)
    #    blob = blob.float()
    #
    #    blob = preprocess_blob(blob)
    #
    #    if self.transform:
    #        blob = self.transform(blob)
    #
    #    return blob, label
    def __getitem__(self, idx):
        img_idx, blob_idx = self.indices[idx]
        label = self.labels[idx]
        if self.version == 1:
            #print("-"*100)
            #print(f"[DEBUG] Using version 1 with in_channel_dist=True")
            #print("-"*100)
            blob, oob = self.blob_helper.get_blob(img_idx, blob_idx, in_channel_dist=True)
        elif self.version == 2:
            #print("-"*100)
            #print(f"[DEBUG] Using version 2 with in_channel_dist=False")
            #print("-"*100)
            blob, oob = self.blob_helper.get_blob(img_idx, blob_idx, in_channel_dist=False)
        elif self.version == 3:
            blob, oob = self.blob_helper.get_blob(img_idx, blob_idx, in_channel_dist=True, binary_mask=True)
        if self.transform:
            blob = self.transform(blob)
        else:
            if isinstance(blob, np.ndarray):
                blob = torch.from_numpy(blob)
        blob = blob.float()

        blob = preprocess_blob(blob)

        return blob, label
    def change_version(self, version):
        #print("-"*100)
        #print(f"[DEBUG] changing to version {version}")
        #print("-"*100)
        self.version = version

def calculate_test_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy