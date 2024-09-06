# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:51:06 2024

@author: sebas
"""

import os
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
import pickle

main_data_folder = "D:\\20240816_Barley_SWIR_micro_germinated\\ref_corrected" 
dataset = HsiDataset(main_data_folder, data_ext='ref')

HSIreader = HsiReader(dataset)

# Define your pixel area threshold here
min_pixel_area = 1000  # Adjust this value as needed
padding = 200000 # increase the bounding box size by x pixels in each direction

for idx in range(len(dataset)):
    
    HSIreader.read_image(idx)
    metadata = HSIreader.current_metadata
    wv = HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hypercube = HSIreader.get_hsi()
    n_samples = 10000
    
    x_idx = np.random.randint(0, hypercube.shape[0], size=n_samples)
    y_idx = np.random.randint(0, hypercube.shape[1], size=n_samples)
    
    spectral_samples = hypercube[x_idx, y_idx, :]

    nb_pca_comp = 3
    pca = PCA(n_components=nb_pca_comp)
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # for i in range(np.shape(pca_loadings)[1]):
    #     plt.figure()
    #     plt.plot(wv, pca_loadings[:, i], default_colors[i])
    #     plt.xlabel("Wavelength (nm)")
    #     plt.ylabel("Reflectance")  
    #     lab = 'PC' + str(i + 1)
    #     plt.title(lab) 
    #     plt.grid()  
    # plt.show(block=False)
    
    score_img = HSIreader.project_pca_scores(pca_loadings)
   
    score_pc_ref = score_img[:, :, 1]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=2)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    labeled_image = label(segmented)
    
    binary_image = labeled_image > 0
    
    filled_binary_image = binary_fill_holes(binary_image)
    
    labeled_image = label(filled_binary_image)
    labeled_image = label(remove_small_objects(labeled_image > 0, min_size=20))
    
    color_image = color_labels(labeled_image)
    
    regions = regionprops(labeled_image)
    print(f"Number of regions found: {len(regions)}")
    object_data = []
    
    for region in regions:
        # Get the object ID (label)
        obj_id = region.label
        
        # Skip the background (label 0)
        if obj_id == 0:
            continue
        
        # Filter out regions based on pixel area
        if region.area < min_pixel_area:
            continue
        
        # Get the centroid coordinates
        centroid = region.centroid
        
        # Get the coordinates of all pixels belonging to this object
        pixel_coords = np.array(region.coords)  # (num_pixels, 2) array
        
        # Get the original bounding box of the region
        min_row, min_col, max_row, max_col = region.bbox
        
        # Expand the bounding box by padding, ensuring it stays within the image boundaries
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(hypercube.shape[0], max_row + padding)
        max_col = min(hypercube.shape[1], max_col + padding)
        
        # Store in dictionary
        object_data.append({
            'id': obj_id,
            'centroid': centroid,
            'pixels': pixel_coords,
            'bbox': (min_row, min_col, max_row, max_col)  # Store the expanded bounding box
        })

    object_data.sort(key=lambda x: (x['centroid'][1], x['centroid'][0]))

    # Convert list to dictionary with ordered keys
    ordered_object_data = {obj['id']: obj for obj in object_data}

    # Create the object folder if it doesn't exist
    object_folder = os.path.join(main_data_folder, 'object')
    os.makedirs(object_folder, exist_ok=True)

    # Create a filename based on the index (e.g., 'object_0001.pkl')
    image_file_name = f'object_{idx:04d}.pkl'

    # Full path for saving the file
    file_path = os.path.join(object_folder, image_file_name)

    # Save the ordered dictionary to a file using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(ordered_object_data, f)
        
#%%

import os
import pickle
import scipy.io as sio
import numpy as np

# Load the saved object data from the pickle file
main_data_folder = "D:\\SWIR_sub_imtest" 
object_folder = os.path.join(main_data_folder, 'object')
object_file = 'object_0000.pkl'
file_path = os.path.join(object_folder, object_file)

with open(file_path, 'rb') as f:
    ordered_object_data = pickle.load(f)

# Prepare data for MATLAB
matlab_data = {}

for obj_id, obj_data in ordered_object_data.items():
    centroid = np.array(obj_data['centroid'])
    pixel_coords = np.array(obj_data['pixels'])
    
    # Calculate bounding box (assuming pixel_coords contains (row, col) pairs)
    min_row, min_col = np.min(pixel_coords, axis=0)
    max_row, max_col = np.max(pixel_coords, axis=0)
    
    # Store in a dictionary compatible with MATLAB structure
    matlab_data[f'object_{obj_id}'] = {
        'centroid': centroid,
        'pixels': pixel_coords,
        'bbox': [min_row, min_col, max_row, max_col]
    }

# Save the data as a .mat file
mat_file_path = os.path.join(object_folder, 'object_0000.mat')
sio.savemat(mat_file_path, matlab_data)


    
