from collections import defaultdict
from skimage.measure import label, regionprops, find_contours
import matplotlib.pyplot as plt
import numpy as np

def plot_mask_with_all_contours(mask2D, contours_all, title="Mask with Contours"):
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


import numpy as np

def normalize_with_cutoffs(data, lower_pct=1, upper_pct=99):
    """
    Normalize data to [0, 1] range after applying percentile cutoffs.

    Parameters:
        data (array-like): Input data.
        lower_pct (float): Lower percentile for clipping.
        upper_pct (float): Upper percentile for clipping.

    Returns:
        np.ndarray: Normalized data clipped and scaled between 0 and 1.
    """
    data = np.asarray(data)
    lower = np.percentile(data, lower_pct)
    upper = np.percentile(data, upper_pct)
    clipped = np.clip(data, lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    return normalized
