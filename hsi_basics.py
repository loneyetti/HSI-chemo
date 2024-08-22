import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *


main_data_folder = "D:\\HSI data\\Barley_ground_30cm_SWIR"     
dataset =HsiDataset(main_data_folder,data_ext='hyspex')

HSIreader = HsiReader(dataset)

for idx in range(len(dataset)):
    
    HSIreader.read_image(idx)
    metadata = HSIreader.current_metadata
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    
    rgb= HSIreader.get_rgb()
    plt.figure() 
    plt.imshow(rgb)
    plt.title("Pseudo RGB Image")
    plt.axis('off')
    plt.show()

    HSIreader.get_spectrum()
    
    hypercube=HSIreader.get_hsi()
    
    
    
    
    
    HSIreader.clear_cache()
    