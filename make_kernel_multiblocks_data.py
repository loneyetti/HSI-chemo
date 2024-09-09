import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *
import pickle
import os

from sklearn.decomposition import PCA

def pca__masked(subimage, mask, n_components=3):
    # Extract masked data from subimage
    mask = mask.astype(bool)
    masked_data = subimage[mask]
    
    # Flatten the masked data to 2D (pixels x bands)
    masked_data_flattened = masked_data.reshape(-1, subimage.shape[2])
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_scores  = pca.fit_transform(masked_data_flattened)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    height, width, _ = subimage.shape
    pca_scores_img = np.zeros((height, width, n_components), dtype=np.float32)
    mask_flat = mask.flatten()
    
    # Place PCA scores back into the PCA score image
    pca_scores_img_flat = pca_scores_img.reshape(-1, n_components)
    pca_scores_img_flat[mask_flat] = pca_scores
    
    # Reshape back to the original subimage shape with PCA component depth
    pca_scores_img = pca_scores_img.reshape(height, width, n_components)
    
    
    return pca_scores_img, pca_loadings

def create_binary_mask(bbox, pixel_coords):
    # Get bounding box parameters
    min_row, min_col, max_row, max_col = bbox
    
    # Calculate the size of the mask without extra rows or columns
    height = max_row - min_row
    width = max_col - min_col
    # Create an empty binary mask with the size of the bounding box
    mask = np.zeros((height, width), dtype=np.bool_)
    
    # Set the corresponding pixels in the mask to 1
    for row, col in pixel_coords:
        if min_row <= row < max_row and min_col <= col < max_col:
            mask[row - min_row, col - min_col] = 1
    
    return mask

main_data_folder = "D:\HSI data\micro_SWIR"   
kernel_folder = os.path.join(main_data_folder, 'Kernels_objects')

dataset =HsiDataset(main_data_folder,data_ext='ref')
HSIreader = HsiReader(dataset)

for idx in range(len(dataset)):
    HSIreader.read_image(idx)
    metadata = HSIreader.current_metadata
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    base_name = os.path.basename(HSIreader.dataset[idx]['data'])
    base_name = os.path.splitext(base_name)[0]
    
    kernels_file_path = os.path.join(kernel_folder, f'{base_name}.pkl')
    
    if os.path.exists(kernels_file_path):
        # Load object data
        with open(kernels_file_path, 'rb') as file:
            kernel_data = pickle.load(file)
            
    for kernel_id, kernel_info in kernel_data.items():
         print(kernel_id)
         pixel_coords = kernel_info['pixels']
         min_row = np.min(pixel_coords[:, 0])
         max_row = np.max(pixel_coords[:, 0])
         min_col = np.min(pixel_coords[:, 1])
         max_col = np.max(pixel_coords[:, 1])
         
         bbox = (min_row, min_col, max_row, max_col)    
            # Read the subimage using the bounding box
         subimage = HSIreader.read_subimage(bbox)
         bm = create_binary_mask(bbox, pixel_coords)
         avg_spectrum = np.mean(subimage[bm == 1], axis=0)
         pca_scores_img, pca_loadings =pca__masked(subimage,mask=bm,n_components=3)
    
       