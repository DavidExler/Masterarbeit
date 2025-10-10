from collections import defaultdict
from skimage.measure import label, regionprops, find_contours
import matplotlib.pyplot as plt
import numpy as np

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