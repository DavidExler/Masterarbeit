from collections import defaultdict
from skimage.measure import label, regionprops, find_contours
import numpy as np
from scipy.fft import fft
from Labeling_App.helpers.visualization_helper import extract_contours_from_mask

def extract_features_and_contours(mask2D):
    contours_raw = extract_contours_from_mask(mask2D)

    contours_by_label = {}
    for item in contours_raw:
        label_id = item['label']
        contour = item['contour']
        contours_by_label.setdefault(label_id, {'contours': [] , 'features': []})
        contours_by_label[label_id]['contours'].append(contour)

    for label_id, data in contours_by_label.items():
        contour_list = data['contours']

        # Create binary mask for this label
        binary_mask = (mask2D == label_id).astype(np.uint8)
        labeled = label(binary_mask)

        # Initialize accumulators
        total_area = 0
        total_perimeter = 0
        total_equiv_diameter = 0
        #centroid_x_sum = 0
        #centroid_y_sum = 0
        #region_count = 0

        for prop in regionprops(labeled):
            total_area += prop.area
            total_perimeter += prop.perimeter
            total_equiv_diameter += prop.equivalent_diameter
            #centroid_x_sum += prop.centroid[0]
            #centroid_y_sum += prop.centroid[1]
            #region_count += 1

        # Average centroid across all regions
        #centroid = (centroid_x_sum / region_count, centroid_y_sum / region_count) if region_count > 0 else (0,0)

        # Combine all contours for Fourier
        combined_contour = np.vstack(contour_list)
        complex_contour = combined_contour[:, 1] + 1j * combined_contour[:, 0]
        fourier = fft(complex_contour)
        fourier_abs = np.abs(fourier[:10])  # First 10 coefficients
        
        if not len(fourier_abs) == 10:
            fourier_abs = np.pad(fourier_abs, (0, 10 - len(fourier_abs)), mode='constant')
        if not len(fourier_abs) == 10:
            print(f"{label_id}")
        #print
        # Compose the feature list
        features = [
            total_area,
            total_perimeter,
            total_equiv_diameter,
            #centroid,
            *fourier_abs.tolist()
        ]
            
        # Optionally remove raw contours if you only want combined
        del contours_by_label[label_id]['contours']

        # Save combined contour and features inside dict
        contours_by_label[label_id]['contours'] = combined_contour
        contours_by_label[label_id]['features'] = features


    return contours_by_label


from sklearn.cluster import KMeans
def cluster_features(contours_with_features, k, randomstate=42):
    labels = list(contours_with_features.keys())
    
    # Extract features into a list in label order
    feature_list = []
    for l in labels:
        feature_list.append(contours_with_features[l]['features'])
    print(len(feature_list))
    print(len(labels))
    # Convert to numpy array
    X = np.array(feature_list)
    
    # Run KMeans
    kmeans = KMeans(n_clusters=k, random_state=randomstate).fit(X)
    classes = kmeans.labels_
    
    # Copy original dict and add 'class' for each label
    contours_with_classes = {}
    for label, cls in zip(labels, classes):
        contours_with_classes[label] = dict(contours_with_features[label])  # shallow copy
        contours_with_classes[label]['class'] = int(cls)
        
    return contours_with_classes