# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:05:08 2024

@author: sebas
"""

import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *
import pickle
import os

from sklearn.decomposition import PCA
import numpy as np

def pca__masked(subimage, mask, n_components=3):
    mask = mask.astype(bool)
    masked_data = subimage[mask]
    masked_data_flattened = masked_data.reshape(-1, subimage.shape[2])
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(masked_data_flattened)
    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    height, width, _ = subimage.shape
    pca_scores_img = np.zeros((height, width, n_components), dtype=np.float32)
    mask_flat = mask.flatten()

    pca_scores_img_flat = pca_scores_img.reshape(-1, n_components)
    pca_scores_img_flat[mask_flat] = pca_scores

    return pca_scores_img.reshape(height, width, n_components), pca_loadings

def create_binary_mask(bbox, pixel_coords):
    min_row, min_col, max_row, max_col = bbox
    height = max_row - min_row
    width = max_col - min_col
    mask = np.zeros((height, width), dtype=np.bool_)
    
    for row, col in pixel_coords:
        if min_row <= row < max_row and min_col <= col < max_col:
            mask[row - min_row, col - min_col] = 1
    
    return mask

def process_images(main_data_folder, kernel_folder):
    dataset = HsiDataset(main_data_folder, data_ext='ref')
    HSIreader = HsiReader(dataset)

    all_avg_spectra = []

    for idx in range(len(dataset)):
        HSIreader.read_image(idx)
        metadata = HSIreader.current_metadata
        wv = HSIreader.get_wavelength()
        wv = [int(l) for l in wv]

        base_name = os.path.basename(HSIreader.dataset[idx]['data'])
        base_name = os.path.splitext(base_name)[0]

        kernels_file_path = os.path.join(kernel_folder, f'{base_name}.pkl')

        if os.path.exists(kernels_file_path):
            # Load kernel (mask) data
            with open(kernels_file_path, 'rb') as file:
                kernel_data = pickle.load(file)

            for kernel_id, kernel_info in kernel_data.items():
                pixel_coords = kernel_info['pixels']
                min_row = np.min(pixel_coords[:, 0])
                max_row = np.max(pixel_coords[:, 0])
                min_col = np.min(pixel_coords[:, 1])
                max_col = np.max(pixel_coords[:, 1])

                # Check if the bounding box is valid
                if min_row >= max_row or min_col >= max_col:
                    print(f"Invalid bounding box for kernel {kernel_id} in image {base_name}. Skipping...")
                    continue

                bbox = (min_row, min_col, max_row, max_col)
                subimage = HSIreader.read_subimage(bbox)
                bm = create_binary_mask(bbox, pixel_coords)

                # Check if subimage or mask is empty, and skip if so
                if subimage.size == 0 or bm.size == 0:
                    print(f"Empty subimage or mask for kernel {kernel_id} in image {base_name}. Skipping...")
                    continue

                avg_spectrum = np.mean(subimage[bm == 1], axis=0)
                
                # Store the average spectrum of this kernel (masked region)
                all_avg_spectra.append(avg_spectrum)

    return np.array(all_avg_spectra)


# Main folder paths
main_data_folder = "D:\\SWIR_Barley_No_damage_matched"
kernel_folder = os.path.join(main_data_folder, 'Kernels_objects')

# Process the images and retrieve average spectra for all images
all_avg_spectra = process_images(main_data_folder, kernel_folder)

# Save the results as both NumPy and MATLAB files
import scipy.io as io

np.save(os.path.join(main_data_folder, 'average_spectra.npy'), all_avg_spectra)
io.savemat(os.path.join(main_data_folder, 'average_spectra.mat'), {"average_spectra": all_avg_spectra})

print("Processing complete. Average spectra have been saved.")

