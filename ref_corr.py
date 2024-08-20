import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA

main_data_folder = "D:\\HSI data\\Barley_ground_30cm_SWIR"     
dataset =HsiDataset(main_data_folder,data_ext='hyspex')

HSIreader = HsiReader(dataset)

for idx in range(len(dataset)):
    
    HSIreader.read_image(idx)
    metadata = HSIreader.current_metadata
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hypercube= HSIreader.get_hsi()
    n_samples = 10000
    
    x_idx = np.random.randint(0, hypercube.shape[0], size=n_samples)
    y_idx = np.random.randint(0, hypercube.shape[1], size=n_samples)
    
    spectral_samples = hypercube[x_idx, y_idx, :]
    # spectral_samples = hypercube[::25,::25]
    
    nb_pca_comp =3
    pca = PCA(n_components=nb_pca_comp)
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    
    plt.figure()
    plt.plot(wv,pca_loadings)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")  
    plt.title('PCA loadings') 
    plt.grid()  

    legends=['PC1','PC2','PC3']
    plt.legend(legends,loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=4) 
    plt.show()

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(np.shape(pca_loadings)[1]):
        plt.figure()
        plt.plot(wv,pca_loadings[:,i],default_colors[i])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")  
        lab= 'PC'+str(i+1)
        plt.title(lab) 
        plt.grid()  
    plt.show()
    
    HSIreader.clear_cache()