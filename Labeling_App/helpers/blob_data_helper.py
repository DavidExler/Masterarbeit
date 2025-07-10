import pickle
import numpy as np
#from helpers.visualization_helper import normalize_with_cutoffs
# load the masks
masks3D_20xRenamed = []
#with open('helpers/masks3D_CELLPOSE_RUN_1.pkl', 'rb') as f:
#    masks3D_20xRenamed = pickle.load(f)
with open('helpers/masks3D_CELLPOSE_RUN_1_Quadrants.pkl', 'rb') as f:
    masks3D_20xRenamed = pickle.load(f)
blobs_per_image = [len(np.unique(mask)) for mask in masks3D_20xRenamed]

images3D_20xRenamed_full = []
with open('helpers/imgs_20xRenamed.pkl', 'rb') as f:
    images3D_20xRenamed_full = pickle.load(f)

images3D_20xRenamed = []
for im in images3D_20xRenamed_full:
    images3D_20xRenamed.append(im[:,0,:,:])
#li = len(images3D_20xRenamed)

normalized_channels = []
with open('helpers/normalized_channels.pkl', 'rb') as f:
    normalized_channels = pickle.load(f)


def normalize_channel(data):
    lower = np.percentile(data, 1)
    upper = np.percentile(data, 99)
    data = np.clip(data, lower, upper)
    return ((data - lower) / (upper - lower) * 255).astype(np.uint8)

with open('helpers/quadrant_cutoffs.pkl', 'rb') as f:
    cutoffs = pickle.load(f)

class BlobDataHelper:
    def __init__(self):
        self.li = len(normalized_channels)
        self.last_image = 0
        self.last_blob = 1
        self.image = np.stack([images3D_20xRenamed[self.last_image]] * 3, axis=-1)
        self.mask = masks3D_20xRenamed[self.last_image]
        self.lum = len(np.unique(self.mask))
        self.blob = self.mask == 1
        self.abs_coords = (0,0,0,0)
        self.selected_channel = []

    def get_blob(self, image_index, blob_index, selected_channels, start_size = 45, offset=2):
        """
        Extract a Region containing the cell specified by the blob_index in the image specified by the image_index.
        This function uses preloaded, caced images and masks, that should be available at the relative paths:
            - save_data/masks/20xRenamed/masks3D_CELLPOSE_RUN_1.pkl 
            - save_data/3D_images_Renamed/imgs_20xRenamed.pkl


        Parameters:
            self: ignore
            image_index: Index of the image to extract the cell from.
            blob_index: Value of the 3D mask associated with a specific cell.
            start_size: Minimal size of the region to  extract. Relative size of cells is imperative, thus every extracted cell region begins at this size.
            offset: Size offset for the region to extract. The exracted region grows by this number of pixels in every direction.

        Returns:
            bbox: The Extracted Region of the original image containing the relevant cell.
            blob_index: Over-/Underflow protected blob index.
            image_index: Over-/Underflow protected image index.
            edge_blob: bool that specifies, wether or not the region touches the image border.
            inside_box: (x_min , x_max, y_min , y_max): Bounding box of the Cell within the extracted region with coordinates relative to the extracted region.
        """
        selected_channels = sorted(selected_channels)
        for ch in selected_channels:
            print(f"[DEBUG] Channel {ch}")
        # Load new image only if necessary
        if not self.last_image == image_index or self.selected_channel != selected_channels:
            # image index overflow protect
            image_index = image_index % self.li
            # image index underflow protect
            if image_index < 0:
                image_index = self.li - image_index

            self.selected_channel = selected_channels
            full_image = normalized_channels[image_index]
            self.select_channel(full_image, selected_channels)
            self.mask = masks3D_20xRenamed[image_index]
            self.lum = len(np.unique(self.mask))
            print(f"image changed from {self.last_image} to {image_index} -> loaded img of shape {self.image.shape} with additional channels {selected_channels} and mask of shape {self.mask.shape} which has {self.lum} indices. The selected blob is {blob_index}")
        
        else:
            print(f"still at {self.last_image} == {image_index} -> Still img of shape {self.image.shape} with additional channels {selected_channels} and mask of shape {self.mask.shape} which has {self.lum} indices. The selected blob is {blob_index}")
        self.last_image = image_index

        # blob index overflow protect
        if blob_index == self.lum:
            blob_index = 1
        # blob index underflow protect
        if blob_index < 0:
            blob_index = self.lum + blob_index
        if blob_index == 0:
            blob_index = self.lum - 1
        self.blob = self.mask == blob_index
        self.last_blob = blob_index
        
        if not np.any(self.blob):
            return None, None  

        # Get coordinates where mask is True
        coords = np.argwhere(self.blob)
        z_min, z_max = coords[:, 0].min(), coords[:, 0].max()

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
            edge_blob = True
        if x_max > self.mask.shape[1]:
            x_min -= x_max - self.mask.shape[1]
            x_max = self.mask.shape[1]
            x_max_abs -= 2
            edge_blob = True
        x_min = max(x_min, 0)
        x_max = min(x_max, self.mask.shape[1])

        # Correct for boundary clipping in y
        if y_min < 0:
            y_max += -y_min
            y_min = 0
            y_min_abs += 2
            edge_blob = True
        if y_max > self.mask.shape[2]:
            y_min -= y_max - self.mask.shape[2]
            y_max = self.mask.shape[2]
            y_max_abs -= 2
            edge_blob = True
        y_min = max(y_min, 0)
        y_max = min(y_max, self.mask.shape[2])

        #min (gesamte return bbox) - min_abs (eigentliche minimalste bbox) = Abstand inside_box_min von relativer 0
        #min (gesamte return bbox) - max_abs (eigentliche minimalste bbox) = Abstand inside_box_max von relativer 0
        inside_box = (x_min_abs - 2 - x_min, x_max_abs + 2 - x_min, y_min_abs - 2 - y_min , y_max_abs + 2 - y_min)
        self.abs_coords = (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max)

        bbox = self.image[z_min:z_max, x_min:x_max, y_min:y_max, :]
        return bbox, blob_index, image_index, edge_blob, inside_box
    
    def get_fullscreen_for_current_blob(self, z_offset, offset=5):
        quadrant = 1
        #quadrant_cutoffs = cutoffs[self.last_image]
        #if quadrant_cutoffs['Q1_max'] <= self.last_blob:
        #    quadrant += 1
        #    if quadrant_cutoffs['Q2_max'] <= self.last_blob:
        #        quadrant += 1
        #        if quadrant_cutoffs['Q3_max'] <= self.last_blob:
        #            quadrant += 1
        #print(f"[DEBUG] cutoffs of picture {self.last_image} with blob {self.last_blob}: {quadrant_cutoffs} - Quadrant: {quadrant}")

        (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max) = self.abs_coords
        print(f"[DEBUG] the abs coords are {(x_min_abs, x_max_abs, y_min_abs, y_max_abs)}")
        
        x_center = (x_min_abs + x_max_abs) / 2
        y_center = (y_min_abs + y_max_abs) / 2
        x_half = self.mask.shape[1] / 2
        y_half = self.mask.shape[2] / 2
        if x_center <= x_half and y_center <= y_half:
            quadrant = 1
        elif x_center > x_half and y_center <= y_half:
            quadrant = 2
        elif x_center <= x_half and y_center > y_half:
            quadrant = 3
        else:
            quadrant = 4

        z = z_min + z_offset
        shape1 = self.image.shape[1]
        shape2 = self.image.shape[2]
        if offset > shape1 / 2:
            offset = 5
        if quadrant == 1:
            fullscreen = self.image[z][0:int(shape1/2) + offset,0:int(shape2/2) + offset]
            print(f"[DEBUG] Getting Quadrant: {quadrant} so x from 0 to {int(shape1/2) + offset} and y from 0 to {int(shape2/2) + offset}")
        elif quadrant == 2:
            fullscreen = self.image[z][int(shape1/2) - offset:shape1,0:int(shape2/2) + offset]
            print(f"[DEBUG] Getting Quadrant: {quadrant} so x from {int(shape1/2) - offset} to {shape1} and y from 0 to {int(shape2/2) + offset}")
            x_min_abs -= (int(shape1/2) - offset)
            x_max_abs -= (int(shape1/2) - offset)
        elif quadrant == 3:
            fullscreen = self.image[z][0:int(shape1/2) + offset,int(shape2/2) - offset:shape2]
            print(f"[DEBUG] Getting Quadrant: {quadrant} so x from 0 to {int(shape1/2) + offset} and y from {int(shape2/2) - offset} to {shape2}")
            y_min_abs -= (int(shape2/2) - offset)
            y_max_abs -= (int(shape2/2) - offset)
        else:
            fullscreen = self.image[z][int(shape1/2) - offset:shape1,int(shape2/2) - offset:shape2]
            print(f"[DEBUG] Getting Quadrant: {quadrant} so x from {int(shape1/2) - offset} to {shape1} and y from {int(shape2/2) - offset} to {shape2}")
            x_min_abs -= (int(shape1/2) - offset)
            x_max_abs -= (int(shape1/2) - offset)
            y_min_abs -= (int(shape2/2) - offset)
            y_max_abs -= (int(shape2/2) - offset)
        print(f"[DEBUG] now the abs coords become {(x_min_abs, x_max_abs, y_min_abs, y_max_abs)}")
        

        return fullscreen, (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max)

    def select_channel(self, full_img, selected_channels):
        print(f"image shape: {self.image.shape}, full_img shape: {full_img.shape}")
        # Reallocate only if needed
        if self.image.shape != (*full_img.shape[0:3], 3):
            self.image = np.zeros((*full_img.shape[0:3], 3), dtype=np.float32)
        else:
            self.image[:] = 0  # Reuse memory when possible

        self.image[:, :, :, 0] = normalize_channel(full_img[:, :, :, 0])
        if 1 in selected_channels:
            self.image[:, :, :, 1] = normalize_channel(full_img[:, :, :, 1])
        if 2 in selected_channels:
            self.image[:, :, :, 2] = normalize_channel(full_img[:, :, :, 2])
        if 3 in selected_channels:
            self.image[:, :, :, 2] = normalize_channel(full_img[:, :, :, 3])
        if 4 in selected_channels:
            self.image[:, :, :, 2] = normalize_channel(full_img[:, :, :, 3])
        return self.image




def get_next_undef(label_store):
    """
    Find the next unlabeled blob in the preloaded, cached collection of images.

    Parameters:
        label_store (dict): Dictionary containing the label_store.json storing labeled blobs per image with keys like "img0", "img1", etc.

    Returns:
        tuple: (img_idx, blob_idx) where:
            img_idx (int): Index of the image containing the next unlabeled blob.
            blob_idx (int): Index of the unlabeled blob within the image.
        Returns (None, None) if all blobs are labeled.
    """

    num_images = len(blobs_per_image)
    for img_idx in range(num_images):
        img_key = f"img{img_idx}"
        labeled_blobs = label_store.get(img_key, {})
        total_blobs = blobs_per_image[img_idx]
        for blob_idx in range(1, total_blobs):# + 1):
            if str(blob_idx) not in labeled_blobs:
                return img_idx, blob_idx
    return None, None  


def sort_cells_by_position():
    new_masks3D = []
    all_mappings = []  

    for i, mask in enumerate(masks3D_20xRenamed):
        print(f"----- Processing image {i} -----")
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]  # Skip background

        coords_by_label = {}
        for label in unique_labels:
            if label % 100 == 0:
                print(f"label {label} von {len(unique_labels)}")
            coords = np.argwhere(mask == label)
            x_center = (coords[:, 1].min() + coords[:, 1].max()) / 2
            y_center = (coords[:, 2].min() + coords[:, 2].max()) / 2
            coords_by_label[label] = (x_center, y_center)

        x_half = mask.shape[1] / 2
        y_half = mask.shape[2] / 2

        # Sort labels into quadrants
        quadrants = {1: [], 2: [], 3: [], 4: []}
        for label, (x, y) in coords_by_label.items():
            if x <= x_half and y <= y_half:
                quadrants[1].append(label)
            elif x > x_half and y <= y_half:
                quadrants[2].append(label)
            elif x <= x_half and y > y_half:
                quadrants[3].append(label)
            else:
                quadrants[4].append(label)

        new_mask = np.zeros_like(mask, dtype=np.int32)
        label_counter = 1
        mapping = []

        for q in [1, 2, 3, 4]:
            for old_label in sorted(quadrants[q]):  # optional sorting within quadrant
                new_mask[mask == old_label] = label_counter
                mapping.append((old_label, label_counter))
                label_counter += 1

        new_masks3D.append(new_mask)
        all_mappings.append(mapping)

        with open('masks3D_CELLPOSE_RUN_1_Quadrants.pkl', 'wb') as handle:
            pickle.dump(new_masks3D, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('masks3D_CELLPOSE_RUN_1_QuadrantMapping.pkl', 'wb') as handle:
            pickle.dump(all_mappings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return new_masks3D, all_mappings

def get_quadrant_cutoffs(mapping_msks):
    """
    For each image, return a dict of cutoff label indices per quadrant.
    Cutoffs are returned as the *maximum* label in each quadrant.
    """
    quadrant_cutoffs = []

    for image_idx, mapping in enumerate(mapping_msks):
        # mapping: list of tuples (old_label, new_label)
        num_labels = len(mapping)
        cutoffs = {}

        # Labels were assigned in order: Q1 → Q2 → Q3 → Q4
        labels_per_quad = num_labels // 4
        extra = num_labels % 4  # in case not divisible evenly

        counts = [labels_per_quad + (1 if i < extra else 0) for i in range(4)]
        cumulative = np.cumsum(counts)

        cutoffs = {
            'Q1_max': cumulative[0],
            'Q2_max': cumulative[1],
            'Q3_max': cumulative[2],
            'Q4_max': cumulative[3],
        }

        quadrant_cutoffs.append(cutoffs)

    return quadrant_cutoffs
