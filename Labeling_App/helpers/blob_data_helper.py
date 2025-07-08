import pickle
import numpy as np
# load the masks
masks3D_20xRenamed = []
with open('helpers/masks3D_CELLPOSE_RUN_1.pkl', 'rb') as f:
    masks3D_20xRenamed = pickle.load(f)
blobs_per_image = [len(np.unique(mask)) for mask in masks3D_20xRenamed]

images3D_20xRenamed_full = []
with open('helpers/imgs_20xRenamed.pkl', 'rb') as f:
    images3D_20xRenamed_full = pickle.load(f)

images3D_20xRenamed = []
for im in images3D_20xRenamed_full:
    images3D_20xRenamed.append(im[:,0,:,:])
#li = len(images3D_20xRenamed)

class BlobDataHelper:
    def __init__(self):
        self.li = len(images3D_20xRenamed)
        self.last_image = 0
        self.image = images3D_20xRenamed[self.last_image]
        self.mask = masks3D_20xRenamed[self.last_image]
        self.lum = len(np.unique(self.mask))
        self.blob = self.mask == 1
        self.abs_coords = (0,0,0,0)

    def get_blob(self, image_index, blob_index, start_size = 45, offset=2):
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
            print(f"image changed from {self.last_image} to {image_index} -> loaded img of shape {self.image.shape} and mask of shape {self.mask.shape} which has {self.lum} indices. The selected blob is {blob_index}")
        
        else:
            print(f"still at {self.last_image} == {image_index} -> Still img of shape {self.image.shape} and mask of shape {self.mask.shape} which has {self.lum} indices. . The selected blob is {blob_index}")
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
        if x_max > self.image.shape[1]:
            x_min -= x_max - self.image.shape[1]
            x_max = self.image.shape[1]
            x_max_abs -= 2
            edge_blob = True
        x_min = max(x_min, 0)
        x_max = min(x_max, self.image.shape[1])

        # Correct for boundary clipping in y
        if y_min < 0:
            y_max += -y_min
            y_min = 0
            y_min_abs += 2
            edge_blob = True
        if y_max > self.image.shape[2]:
            y_min -= y_max - self.image.shape[2]
            y_max = self.image.shape[2]
            y_max_abs -= 2
            edge_blob = True
        y_min = max(y_min, 0)
        y_max = min(y_max, self.image.shape[2])

        #min (gesamte return bbox) - min_abs (eigentliche minimalste bbox) = Abstand inside_box_min von relativer 0
        #min (gesamte return bbox) - max_abs (eigentliche minimalste bbox) = Abstand inside_box_max von relativer 0
        inside_box = (x_min_abs - 2 - x_min, x_max_abs + 2 - x_min, y_min_abs - 2 - y_min , y_max_abs + 2 - y_min)
        self.abs_coords = (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max)

        bbox = self.image[z_min:z_max, x_min:x_max, y_min:y_max]
        return bbox, blob_index, image_index, edge_blob, inside_box
    
    def get_fullscreen_for_current_blob(self, z_offset, offset=0):
        (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max) = self.abs_coords
        z = z_min + z_offset
        fullscreen = self.image[z]

        return fullscreen, (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max)


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

