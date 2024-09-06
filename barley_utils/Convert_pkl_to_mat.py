# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:14:16 2024

@author: sebas
"""

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


