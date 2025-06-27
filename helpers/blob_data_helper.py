import pickle
import numpy as np
# load the masks
masks3D_20xRenamed = []
with open('save_data/masks/20xRenamed/masks3D_CELLPOSE_RUN_1.pkl', 'rb') as f:
    masks3D_20xRenamed = pickle.load(f)
    
images3D_20xRenamed_full = []
with open('save_data/3D_images_Renamed/imgs_20xRenamed.pkl', 'rb') as f:
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

    def get_blob(self, image_index, blob_index, offset=5):
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
            print(f"image changed from {self.last_image} to {image_index} -> loaded img of shape {self.image.shape} and mask of shape {self.mask.shape} which has {self.lum} indices.")
        else:
            print(f"still at {self.last_image} == {image_index} -> Still img of shape {self.image.shape} and mask of shape {self.mask.shape} which has {self.lum} indices.")
        self.last_image = image_index

        # blob index overflow protect
        blob_index = blob_index % self.lum
        # blob index underflow protect
        if blob_index < 0:
            blob_index = self.lum - blob_index
        self.blob = self.mask == blob_index

        if not np.any(self.blob):
            return None, None  

        # Get coordinates where mask is True
        coords = np.argwhere(self.blob)
        z_min, z_max = coords[:, 0].min(), coords[:, 0].max()
        x_min, x_max = max(0, coords[:, 1].min() - offset), min(self.image.shape[1], coords[:, 1].max() + offset + 1)
        y_min, y_max = max(0, coords[:, 2].min() - offset), min(self.image.shape[2], coords[:, 2].max() + offset + 1)

        edge_blob = x_min == 0 or x_max == self.image.shape[1] or y_min == 0 or y_max == self.image.shape[2]

        # absolute_coords = {'z_min': z_min, 'z_max': z_max, 'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max}
        bbox = self.image[z_min:z_max, x_min:x_max, y_min:y_max]
        return bbox, blob_index, image_index, edge_blob#, absolute_coords






#last_image = 0
#image = images3D_20xRenamed[last_image]
#mask = masks3D_20xRenamed[last_image]
#lum = len(np.unique(masks3D_20xRenamed[last_image]))
#
#def get_blob(image_index, blob_index, offset=5):
#    #Load new image only if necessary
#    if not last_image == image_index:
#        #image index overflow protect
#        image_index = image_index % li
#        #image index underflow protect
#        if image_index < 0:
#            image_index = li - image_index
#        
#        image = images3D_20xRenamed[image_index]
#        mask = masks3D_20xRenamed[image_index]
#        lum = len(np.unique(mask))
#        print(f"image changed from {last_image} to {image_index} -> loaded img of shape {image.shape} and mask of shape {mask.shape} which has {lum} indices.")
#    else:
#        print(f"still at {last_image} == {image_index}") #-> Still img of shape {image.shape} and mask of shape {mask.shape} which has {lum} indices.")
#    #blob index overflow protect   
#    blob_index = blob_index % lum
#    #blob index underflow protect
#    if blob_index < 0:
#        blob_index = lum - blob_index
#    mask = mask == blob_index
#
#    if not np.any(mask):
#        return None, None  
#
#    # Get coordinates where mask is True
#    coords = np.argwhere(mask)
#    z_min, z_max = coords[:, 0].min(), coords[:, 0].max()
#    x_min, x_max = max(0, coords[:, 1].min() - offset), min(image.shape[1], coords[:, 1].max() + offset + 1)
#    y_min, y_max = max(0, coords[:, 2].min() - offset), min(image.shape[2], coords[:, 2].max() + offset + 1)
#
#    edge_blob = x_min == 0 or x_max == image.shape[1] or y_min == 0 or y_max == image.shape[2]
#
#    #absolute_coords = {'z_min': z_min, 'z_max': z_max, 'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max}
#    bbox = image[z_min:z_max, x_min:x_max, y_min:y_max]
#    return bbox, blob_index, image_index, edge_blob#, absolute_coords
#
# 
# 
#def get_blob_slow(image_index, blob_index, offset=5):
#    #print(len(images3D_20xRenamed))
#    image_index = image_index % len(images3D_20xRenamed)
#    if image_index < 0:
#        image_index = len(np.unique(images3D_20xRenamed_full[image_index])) - image_index
#    image = images3D_20xRenamed[image_index]
#    blob_index = blob_index % len(np.unique(masks3D_20xRenamed[image_index]))
#    #if blob_index == 0:
#    #    blob_index = blob_index + 1
#    if blob_index < 0:
#        blob_index = len(np.unique(masks3D_20xRenamed[image_index])) - blob_index
#    mask = masks3D_20xRenamed[image_index] == blob_index
#    
#    if not np.any(mask):
#        return None, None  
#    
#    # Get coordinates where mask is True
#    z, x, y = np.where(mask)
#    z_min, z_max = np.max([0, z.min()]), np.min([z.max(), image.shape[0]])
#    x_min, x_max = np.max([0, x.min() - offset]), np.min([x.max() + offset + 1, image.shape[1]])
#    y_min, y_max = np.max([0, y.min() - offset]), np.min([y.max() + offset + 1, image.shape[2]])
#    
#    edge_blob =False
#    if (x_min == 0) or (x_max == image.shape[1]) or (y_min == 0) or (y_max == image.shape[2]):
#        print("Rand Blob!")
#        edge_blob = True
#    
#    absolute_coords = {'z_min': z_min, 'z_max': z_max, 'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max':x_max}
#    #print(f"z_min, z_max: {z_min, z_max}, x_min, x_max: {x_min, x_max}, y_min, y_max: {y_min, y_max}")
#    bbox = image[z_min:z_max, x_min:x_max, y_min:y_max]
#    return bbox, blob_index, image_index, edge_blob, absolute_coords
#    
#    
#    
#    