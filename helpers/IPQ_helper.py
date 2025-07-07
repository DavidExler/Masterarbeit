import os
import numpy as np
import json
from pathlib import Path

def calculate_iou(mask1, mask2):
    # Calculate the intersection and union of two binary masks
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def check_iou(mask1, mask2, iou_thresh=0.3, contain_thresh1=0.5, contain_thresh2=0.5):
    """
Checks whether two binary masks are sufficiently overlapping to be considered a match.

A match is determined if:
- The Intersection over Union (IoU) exceeds `iou_thresh`, or
- A significant portion of `mask1` is contained in `mask2` (intersection / area1 > contain_thresh1), or
- A significant portion of `mask2` is contained in `mask1` (intersection / area2 > contain_thresh2).

Args:
    mask1 (np.ndarray): First binary mask.
    mask2 (np.ndarray): Second binary mask.
    iou_thresh (float): Minimum IoU threshold to consider masks as matching.
    contain_thresh1 (float): Minimum containment ratio for mask1 inside mask2.
    contain_thresh2 (float): Minimum containment ratio for mask2 inside mask1.

Returns:
    bool: True if masks are considered a match, False otherwise.
    """

    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return False

    union = np.logical_or(mask1, mask2).sum()
    area1 = mask1.sum()
    area2 = mask2.sum()

    iou = intersection / union if union > 0 else 0.0
    contain1 = intersection / area1 if area1 > 0 else 0.0
    contain2 = intersection / area2 if area2 > 0 else 0.0

    return iou > iou_thresh or contain1 > contain_thresh1 or contain2 > contain_thresh2


def map_labels_to_gt(masks3D, gt3D, name, iou_thresh=0.3, contain_thresh1=0.5, contain_thresh2=0.5):
    """
Maps predicted labels in a 3D mask volume to corresponding ground truth labels based on spatial overlap.

Each non-background predicted label is compared against all ground truth labels. 
A predicted label is mapped to a ground truth label if the overlap satisfies
the `check_iou` criteria (IoU or containment).

The result is saved in a JSON file with the structure:
{
    "0": { "1": [2, 4], ... },  # image index 0: predicted label "1" maps to gt labels 2 and 4
    "1": { ... },
    ...
}

Args:
    masks3D (list of np.ndarray): List of predicted mask slices (2D arrays).
    gt3D (list of np.ndarray): List of ground truth mask slices (2D arrays).
    name (str): Identifier name for the output mapping file.
    iou_thresh (float): IoU threshold for overlap comparison.
    contain_thresh1 (float): Containment threshold for predicted mask inside GT.
    contain_thresh2 (float): Containment threshold for GT mask inside predicted.
    """

    mapping = {}
    filename = f"mapping_{name}.json"
    mapping_path = Path("save_data/LabelsAndIoUs")

    if not mapping_path.exists():
        raise FileNotFoundError("Directory does not exist")

    file_path = mapping_path / filename

    # Try to load existing mapping
    if file_path.exists():
        with open(file_path, 'r') as f:
            mapping = json.load(f)
        processed_indices = set(int(k) for k in mapping.keys())
    else:
        processed_indices = set()

    if len(masks3D) != len(gt3D):
        raise ValueError("Masks and ground truths must contain the same number of images")

    for index in range(len(masks3D)):
        if index in processed_indices:
            print(f"already processed")
            continue  # Skip already processed images


        mask_img = masks3D[index]
        gt_img = gt3D[index]
        all_labels = np.unique(mask_img)
        print(f"---------------------------------------------------------------------\n image index {index} with {len(all_labels)} labels \n---------------------------------------------------------------------")

        mapping[str(index)] = {}

        for label in all_labels:
            if label == 0:
                continue
            if label % 100 == 0:
                print(f"processing {label} of {len(all_labels)}")
            
            mask = (mask_img == label)
            gt_matches = []

            for gt_label in np.unique(gt_img):
                if gt_label == 0:
                    continue

                gt_mask = (gt_img == gt_label)

                if check_iou(mask, gt_mask, iou_thresh, contain_thresh1, contain_thresh2):
                    gt_matches.append(int(gt_label))

            mapping[str(index)][str(label)] = gt_matches

        # Save updated mapping after each image
        with open(file_path, 'w') as f:
            json.dump(mapping, f, indent=2)

def count_repeated_items(best_gt_labels):
    seen = {}
    duplicates = []

    for i, item in enumerate(best_gt_labels):
        if item in seen:
            duplicates.append((i, item))
            if item == 84: 
                print(f"Found {item} at {i}.")
        else:
            seen[item] = i  # store first index where item was seen
    return duplicates, seen 


def calculate_pred_to_gt_iou(mask3D, gt3D, mapping):
    """
    Computes IoU between each predicted label and the union of all its matched GT labels.

    Args:
        mask2D (np.ndarray): Predicted label mask.
        gt2D (np.ndarray): Ground truth label mask.
        mapping (dict): Mapping from predicted labels to lists of matched GT labels.

    Returns:
        dict: Mapping from predicted label (int) to IoU (float).
    """
    iou_per_pred = []
    ctr = 0
    for pred_label_str, matched_gts in mapping.items():
        pred_label = int(pred_label_str)
        pred_mask = (mask3D == pred_label)

        if not matched_gts:
            #iou_per_pred[pred_label] = 0.0
            continue
        
        ctr += 1
        # Combine all matched GT masks
        combined_gt_mask = np.zeros_like(gt3D, dtype=bool)
        for gt_label in matched_gts:
            combined_gt_mask |= (gt3D == gt_label)

        iou = calculate_iou(pred_mask, combined_gt_mask)
        #if iou > 1.0:
        #print(f"iteration: {ctr} - iou: {iou}")
            #raise ValueError(f"iou größer als!: {iou}")
        iou_per_pred.append(iou)
    exprected_length = ctr
    return iou_per_pred, exprected_length



from collections import Counter
def calculate_IPQ(mask3D, gt3D, mapping, k1, k2, k3):#, iou_threshold=0.2):
    """
    Calculate the Injektive Panoptische Qualität (IPQ) for a 3D mask against a ground truth mask.
    Parameters:
    - mask3D: 3D numpy array of predicted masks.
    - gt_mask3D: 3D numpy array of ground truth masks.
    - mapping: 
    """

    # initialize containers
    FP = []
    TP = []
    FN = []
    ious = []
    IoU_sum = 0.0
    matched_gt = set()
    gt_assignment_counts = Counter()

#   ----- Recognition Quality -----
    for pred_label_str, matched_gts in mapping.items():
        pred_label = int(pred_label_str)
        if matched_gts:
            TP.append(pred_label)
            matched_gt.update(matched_gts)
            gt_assignment_counts.update(matched_gts)
        else:
            FP.append(pred_label)

    for gt_label in np.unique(gt3D):
        if gt_label == 0:
            continue
        if gt_label not in matched_gt:
            FN.append(gt_label)

    RQ = len(TP) / (len(TP) + 0.5 * (len(FP) + len(FN))) if (len(TP) + len(FN)) > 0 else 0.0
    RQ = k2 * RQ


#   ----- Segmentation Quality -----
    ious, exprected_length = calculate_pred_to_gt_iou(mask3D, gt3D, mapping)
    #print(ious)
    IoU_sum = sum(ious)
    #ctr = 0
    #for iou_singular in ious:
    #    if iou_singular != 0:
    #        ctr += 1
    #        #print(f"iou: {iou_singular} - ctr: {ctr}, accumulated: {IoU_sum}")
    #        #IoU_sum += iou_singular

    if len(ious) != len(TP) or exprected_length != len(TP):
        print(f"length of ious: {len(ious)} and of TP: {len(TP)}")
        raise ValueError("Something went wrong. TPs must be of the same length as the ious!")
    #print(f"length of ious: {len(ious)} - iou sum: {IoU_sum}")
    SQ = IoU_sum / len(ious)
    SQ = k1 * SQ


#   ----- Injective Quality -----
    gt_labels = np.unique(gt3D)
    counts = np.array([gt_assignment_counts[label] if label != 0 else 0 for label in gt_labels])
    IQ = len(np.unique(gt3D)) / sum(max(count, 1) for count in counts)
    IQ = k3 * IQ

    ipq = RQ * SQ * IQ
    print(f"TP: {len(TP)}, FP: {len(FP)}, FN: {len(FN)}, SQ: {SQ}, RQ: {RQ}, IQ: {IQ}, IPQ: {ipq}")
    return TP, FP, FN, SQ, RQ, IQ, ipq

from skimage import measure, morphology, segmentation, filters
from scipy import ndimage as ndi

def split_large_instances(mask, area_thresh=2000):
    new_mask = np.zeros_like(mask)
    next_label = 1
    ctr = 0
    for label in np.unique(mask):
        if label == 0:
            continue
        binary = mask == label
        area = np.sum(binary)
        if area < area_thresh:
            new_mask[binary] = next_label
            next_label += 1
        else:
            ctr += 1
            # Distance transform
            distance = ndi.distance_transform_edt(binary)
            local_maxi = morphology.local_maxima(distance)
            markers = measure.label(local_maxi)
            labels = segmentation.watershed(-distance, markers, mask=binary)
            for split_label in np.unique(labels):
                if split_label == 0:
                    continue
                new_mask[labels == split_label] = next_label
                next_label += 1
    #print(f"changed {ctr} of {len(np.unique(mask))} labels")
    return new_mask
