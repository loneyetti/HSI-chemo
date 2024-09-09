import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *
import pickle
import os

from sklearn.decomposition import PCA
import numpy as np

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



def GLCM(Img, mask, theta,d=1):
    theta = np.deg2rad(theta)
    
    if theta == 0:
        dy, dx = 0, d
    elif theta == np.pi / 4:  
        dy, dx = d, d
    elif theta == np.pi / 2:  
        dy, dx = d, 0
    else:
        raise ValueError("theta must be 0, 45, or 90 degrees")
    
    if nb_bins is None:
        nb_bins = 16
        
    glcm = np.zeros((nb_bins, nb_bins), dtype=np.float32)
    
    height, width = Img.shape

    for y in range(height - dy):
        for x in range(width - dx):
            if mask[y, x] and mask[y + dy, x + dx]:
                pixel_1 = int(Img[y, x])
                pixel_2 = int(Img[y + dy, x + dx])
                pixel_1 = min(pixel_1, nb_bins - 1)
                pixel_2 = min(pixel_2, nb_bins - 1)
                glcm[pixel_1, pixel_2] += 1
                glcm[pixel_2, pixel_1] += 1 

    # Normalize the GLCM
    glcm /= np.sum(glcm)
    return glcm

def haralick(glcm, feature_names=None):
    

    # Initialize feature dictionary
    features = {}


    i = np.arange(glcm.shape[0])
    j = np.arange(glcm.shape[1])

    # 'ASM'
    if feature_names is None or 'ASM' in feature_names:
        asm = np.sum(glcm ** 2)
        features['ASM'] = asm

    #  'Contrast'
    if feature_names is None or 'Contrast' in feature_names:
        contrast = np.sum((i[:, None] - j) ** 2 * glcm)
        features['Contrast'] = contrast

    #  'Correlation'
    if feature_names is None or 'Correlation' in feature_names:
        mean_i = np.sum(i[:, None] * glcm)
        mean_j = np.sum(j * glcm)
        std_i = np.sqrt(np.sum((i[:, None] - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))
        correlation = np.sum((i[:, None] - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)
        features['Correlation'] = correlation

    #  'Variance'
    if feature_names is None or 'Variance' in feature_names:
        variance = np.sum((i[:, None] - mean_i) ** 2 * glcm)
        features['Variance'] = variance

    #  'IDM'
    if feature_names is None or 'IDM' in feature_names:
        idm = np.sum(1 / (1 + (i[:, None] - j) ** 2) * glcm)
        features['IDM'] = idm

    #  'Sum Average'
    if feature_names is None or 'Sum Average' in feature_names:
        sum_avg = np.sum((i[:, None] + j) * glcm)
        features['Sum Average'] = sum_avg

    #  'Sum Variance'
    if feature_names is None or 'Sum Variance' in feature_names:
        sum_variance = np.sum(((i[:, None] + j) - sum_avg) ** 2 * glcm)
        features['Sum Variance'] = sum_variance

    #  'Sum Entropy'
    if feature_names is None or 'Sum Entropy' in feature_names:
        sum_entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Sum Entropy'] = sum_entropy

    #  'Entropy'
    if feature_names is None or 'Entropy' in feature_names:
        entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Entropy'] = entropy

    #  'Difference Variance'
    if feature_names is None or 'Difference Variance' in feature_names:
        diff_var = np.sum(((i[:, None] - j) ** 2) * glcm)
        features['Difference Variance'] = diff_var

    #  'Difference Entropy'
    if feature_names is None or 'Difference Entropy' in feature_names:
        diff_entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Difference Entropy'] = diff_entropy

    #  'IMC1'
    if feature_names is None or 'IMC1' in feature_names:
        imc1 = (entropy - (np.sum(glcm * np.log2(glcm + np.finfo(float).eps)) -
                           np.sum(np.log2(glcm + np.finfo(float).eps) * glcm))) / np.sqrt(np.sum(glcm))
        features['IMC1'] = imc1

    #  'IMC2'
    if feature_names is None or 'IMC2' in feature_names:
        imc2 = (np.sqrt(np.sum(glcm ** 2)) - np.sum(glcm * np.log2(glcm + np.finfo(float).eps))) / np.sqrt(np.sum(glcm))
        features['IMC2'] = imc2

    return features



main_data_folder = "D:\HSI data\micro_SWIR"   
kernel_folder = os.path.join(main_data_folder, 'Kernels_objects')

dataset =HsiDataset(main_data_folder,data_ext='ref')
HSIreader = HsiReader(dataset)

angles = [0, 45, 90]
features=['ASM', 'Contrast', 'Correlation','entropy'] 
num_features = len(features) * len(angles)       
feature_maps = {
    angle: np.zeros((num_kernels, num_features), dtype=np.float32)
    for angle in angles
}


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
            
    
    num_kernels = len(kernel_data)
    num_features = len(features) * 3  # if 3 angles lkets adjust later      
            
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
         # make score images on the mask for glcm 
         pca_scores_img, pca_loadings =pca__masked(subimage,mask=bm,n_components=3)
    
         for i in range(pca_scores_img.shape[2]):
            component_img = pca_scores_img[:, :, i]
            
            for angle in angles:
                glcm = GLCM(component_img, bm, d=1, theta=angle, nb_bins=32)
                features_values = haralick(glcm, feature_names=features)
                
                feature_map_row = []
                for feature in features:
                    feature_map_row.append(features_values.get(feature, 0))
                
                feature_maps[angle][idx, i*len(features):(i+1)*len(features)] = feature_map_row