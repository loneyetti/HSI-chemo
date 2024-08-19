import spectral.io.envi as envi

import tensorly as tl
import numpy as np

import matplotlib.pyplot as plt

from hsi_utils import *


main_data_folder = "D:\\HSI data\\Barley_ground_30cm_SWIR"     
dataset =HsiDataset(main_data_folder,data_ext='hyspex')

HSIreader = HsiReader(dataset)

for idx in range(len(dataset)):
    
    HSIreader.read_image(idx)
    metadata = HSIreader.current_metadata
    # rgb= HSIreader.get_rgb()
    
    # plt.figure() 
    # plt.imshow(rgb)
    # plt.title("RGB Image")
    # plt.axis('off')
    # plt.show()

    # HSIreader.get_spectrum()
    
    
    # HSIreader.clear_cache()
    