from collections import defaultdict
from skimage.measure import label, regionprops, find_contours
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import json
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def visualizer(in_path, out_path):
    # --- Load all masks and predictions ---
    mask_file = os.path.join(BASE_DIR, 'data', in_path, 'masks.pkl')
    pred_file = os.path.join(BASE_DIR, 'data', in_path, 'predictions.json')
    
    with open(mask_file, "rb") as f:
        masks_list = pickle.load(f)  # list of 3D masks
        print(f"found {len(masks_list)} masks")
    
    with open(pred_file, "r") as f:
        predictions_list = json.load(f)  # list of predictions arrays
        predictions_list = predictions_list["predicted"]
        print(f"found {len(predictions_list)} predictions")

    if len(masks_list) == 0:
        raise RuntimeError("No masks found.")

    # --- Graph 3: middle slice of first mask ---
    first_mask = masks_list[0]
    depth = first_mask.shape[0]
    mask_slice = first_mask[depth//2, :, :]
    mask_fig = px.imshow(mask_slice, color_continuous_scale='gray', title="mittlere Schicht der ersten Maske")

    # --- Graph 2: class distribution across all predictions ---
    class_counts = pd.Series(predictions_list).value_counts().sort_index()
    for i in range(4):
        if i not in class_counts:
            class_counts[i] = 0
    class_counts = class_counts.sort_index()
    print(class_counts)
    df_classes = pd.DataFrame({"Class": [f"Class {i}" for i in class_counts.index],
                               "Count": class_counts.values})
    classes_fig = px.bar(df_classes, x="Class", y="Count", title="Segment Class Distribution")

    # --- Graph 1: 3D volumes aggregated over all masks ---
    all_volumes = []
    for mask in masks_list:
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]  # remove background
        volumes = [np.sum(mask == label) for label in unique_labels]
        all_volumes.extend(volumes)
    df_volumes = pd.DataFrame({"Volume (voxels)": all_volumes})
    volumes_fig = px.histogram(df_volumes, x="Volume (voxels)", nbins=20, title="3D Segment Volumes")

    return mask_fig, classes_fig, volumes_fig


def plot_mask_with_all_contours(mask2D, contours_all, title="Mask with Contours"):
    """
    Plot a 2D mask with overlaid contours for each labeled region.

    Parameters:
        mask2D: 2D array representing the segmentation mask.
        contours_all (list of dict): List of contour dictionaries, each with keys:
            - 'label' (int): Label ID of the region.
            - 'contour' (np.ndarray): Contour coordinates as (N, 2) array.
        title (str, optional): Title of the plot. Defaults to "Mask with Contours".

    Returns:
        None: Displays a matplotlib plot showing the mask and contours.
    """

    plt.figure(figsize=(5, 5))
    plt.imshow(mask2D, cmap='gray')
    
    for contour_dict in contours_all:
        if contour_dict is not None:
            contour = contour_dict['contour']
            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

    plt.title(title)
    plt.axis('off')
    plt.show()



def extract_contours_from_mask(mask2D):
    """
    Extract contours for each labeled region in a 2D segmentation mask.

    Parameters:
        mask2D: 2D array of integer labels representing segmented regions.

    Returns:
        list of dict: List where each dict contains:
            - 'label' (int): The label ID for the segmented region.
            - 'contour' (np.ndarray): Coordinates of the contour points for the region.
    """
    contours_all = []

    # Find each unique label (exclude background 0)
    for label_id in np.unique(mask2D):
        if label_id == 0:
            continue
        binary_mask = (mask2D == label_id).astype(np.uint8)
        contours = find_contours(binary_mask, level=0.5)
        
        # Save all contours for this instance
        for contour in contours:
            #contours_all.append(contour)
            contours_all.append({
                'label': label_id,
                'contour': contour
            })

    return contours_all


def normalize_with_cutoffs(data, lower_pct=1, upper_pct=99):
    data = np.asarray(data)
    normalized = np.zeros_like(data, dtype=np.float32)
    for c in range(data.shape[-1]):
        if not np.max(data[..., c]) == 0:
            lower = np.percentile(data[..., c], lower_pct)
            upper = np.percentile(data[..., c], upper_pct)
            clipped = np.clip(data[..., c], lower, upper)
            normalized[..., c] = (clipped - lower) / (upper - lower)
    return normalized



def plot_image_with_clustered_contours_RGB(image, contours_with_classes):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    custom_colors = ["#00FF00", "#0000FF",  "#FFFF00", "#FF00FF", "#00FFFF"] 
    
    for label_id, data in contours_with_classes.items():
        contour = data['contours']
        cls = data['class']
        
        color = custom_colors[cls % len(custom_colors)]  # Cycle colors if classes > len(colors)
        
        plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)
    

    plt.title('Image with Contours. Colors correspond to classes')
    plt.axis('off')
    plt.show()