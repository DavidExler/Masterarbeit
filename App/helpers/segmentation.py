import numpy as np
import os
from cellpose import models, core, io, plot
from pathlib import Path
import tifffile
import pickle
import torch
from helpers.blob_data_helper import read_new_blob_folder

#
# takes from in folder: images (Z,X,Y,C)
# returns to out folder: masks, images (Z,X,Y,C)
#

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
def load_all_tifs(in_folder, out_folder=os.path.join('Demo','Methodenvergleich')):
    folder_path = os.path.join(in_folder)
    print(folder_path)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    # collect all .tif or .tiff files
    tif_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.tif', '.tiff'))]
    print(f"found {len(tif_files)} files")
    if not tif_files:
        raise RuntimeError(f"No .tif files found in folder: {folder_path}")

    # read them all into memory (list of numpy arrays)
    images = [tifffile.imread(os.path.join(folder_path, f)) for f in tif_files]
    with open(os.path.join(BASE_DIR, 'data', 'images.pkl'), 'wb') as f:
        pickle.dump(images, f)  

    with open(os.path.join(BASE_DIR, 'data', out_folder, 'images.pkl'), 'wb') as f:
        pickle.dump(images, f)  

    nuc_images = [img[:,:,:,0] for img in images]
    
    return nuc_images

def segment_folder(in_folder, out_folder):
    path = os.path.join(BASE_DIR, 'data', in_folder)
    print(f"loading tifs from: {path}")
    data = load_all_tifs(path, os.path.join(BASE_DIR, 'data', out_folder))
    print(f"loaded {len(data)} tifs")
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0

    model = models.CellposeModel(gpu=True)
    masks3D, flows3D, styles3D = model.eval(data, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                  normalize={"tile_norm_blocksize": tile_norm_blocksize}, do_3D=True, channel_axis=1, z_axis=0)
    
    num_segments = np.unique(masks3D)
    print(f"found {len(num_segments)} segments")
    path = os.path.join(BASE_DIR, 'data', out_folder)
    if not os.path.isdir(path):
        print(f"path {path} doesnt exist, saving in base directory")
        path = os.path.join(BASE_DIR)
    with open(os.path.join(path, 'masks.pkl'), 'wb') as f:
        pickle.dump(masks3D, f)  
    with open(os.path.join(BASE_DIR, 'data', 'masks.pkl'), 'wb') as f:
        pickle.dump(masks3D, f)  
    read_new_blob_folder(out_folder)
    del(model)
    torch.cuda.empty_cache()
    return num_segments