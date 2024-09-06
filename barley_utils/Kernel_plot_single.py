# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:07:53 2024

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
import os

import numpy as np
import matplotlib.pyplot as plt

# Load the saved object data from the pickle file
main_data_folder = "D:\\20240816_Barley_SWIR_micro_germinated\\ref_corrected"
object_folder = os.path.join(main_data_folder, 'object')
object_file = 'object_0020.pkl'
file_path = os.path.join(object_folder, object_file)

with open(file_path, 'rb') as f:
    ordered_object_data = pickle.load(f)

# Function to plot a sub-image
def plot_sub_image(hypercube, pixel_coords, title=None):
    # Get the bounding box coordinates
    min_row = np.min(pixel_coords[:, 0])
    max_row = np.max(pixel_coords[:, 0])
    min_col = np.min(pixel_coords[:, 1])
    max_col = np.max(pixel_coords[:, 1])

    # Extract the sub-image
    sub_image = hypercube[min_row:max_row+1, min_col:max_col+1, :]

    # Plot the sub-image (you may choose a band to plot, or average over several bands)
    band_index = 50  # Select a band to display (this is just an example)
    plt.figure()
    plt.imshow(sub_image[:, :, band_index], cmap='gray')
    plt.title(title if title else 'Sub-Image')
    plt.axis('off')
    plt.show()

# Load the hyperspectral image data to extract sub-images from
HSIreader = HsiReader(HsiDataset(main_data_folder, data_ext='ref'))
HSIreader.read_image(0)  # Assuming you're working with the first image
hypercube = HSIreader.get_hsi()

# Select a few sub-images to plot
num_images_to_plot = 4
object_ids = list(ordered_object_data.keys())[:num_images_to_plot]  # Adjust to select different objects

for obj_id in object_ids:
    obj_data = ordered_object_data[obj_id]
    pixel_coords = obj_data['pixels']
    plot_sub_image(hypercube, pixel_coords, title=f'Object ID: {obj_id}')
