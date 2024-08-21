import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label
import spectral

main_data_folder = "D:\\HSI data\\Barley_ground_30cm_SWIR"     
dataset =HsiDataset(main_data_folder,data_ext='hyspex')

save_folder = os.path.join(main_data_folder, 'ref_corrected')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

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
    
    
    # plt.figure()
    # plt.plot(wv,pca_loadings)
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Reflectance")  
    # plt.title('PCA loadings') 
    # plt.grid()  

    # legends=['PC1','PC2','PC3']
    # plt.legend(legends,loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #         fancybox=True, shadow=True, ncol=4) 
    # plt.show()

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
    
    score_img = HSIreader.project_pca_scores(pca_loadings)
   
    for s in range(pca_loadings.shape[1]):
        plt.figure()
        plt.imshow(score_img[:,:,s])
        plt.title(f'Score image PC{s+1}')
        plt.axis('off')
        plt.show(block=False)
       
    score_pc_ref = score_img[:,:,0]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=3)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    # plt.figure()
    # plt.imshow(segmented)
    # plt.show(block=False)
    
    labeled_image = label(segmented)
   
    
    # plt.figure()
    # plt.imshow(labeled_image)
    # plt.show(block=False)
    
    binary_image = labeled_image > 0
    
    filled_binary_image = binary_fill_holes(binary_image)
    
    # plt.figure()
    # plt.imshow(filled_binary_image)
    # plt.show(block=False)
    
    labeled_image = label(filled_binary_image)
    labeled_image= label(remove_small_objects(labeled_image > 0, min_size=20))
    
    # plt.figure()
    # plt.imshow(labeled_image)
    # plt.show(block=False)
    
    color_image = color_labels(labeled_image)
        
    plt.figure()
    plt.imshow(color_image)
    plt.title('Color-Mapped Labeled Image')
    plt.axis('off')
    plt.show()
    
    
    reference_mask = labeled_image == 1
    reference_mask=np.repeat(reference_mask[:, :, np.newaxis], hypercube.shape[2], axis=2)
    spectralon = np.where(reference_mask, hypercube, 0)
    avg_spectralon = np.zeros((hypercube.shape[1], hypercube.shape[2]))
    
    non_zero_rows = np.sum(np.any(np.any(spectralon != 0, axis=2), axis=1))
    
    for col in range(hypercube.shape[1]):
        column_data = spectralon[:, col, :]
        non_zero_rows = np.any(column_data != 0, axis=1)
        avg_spectralon[col, :] = np.sum(column_data[non_zero_rows, :], axis=0)
        
        if np.sum(non_zero_rows)>0:
            avg_spectralon[col, :] /= (non_zero_rows[0])
        else:
            avg_spectralon[col, :] = np.nan
        
    normalized_hypercube = np.zeros_like(hypercube, dtype=np.float32)
    for col in range(hypercube.shape[1]):
        # Avoid division by zero
        avg_spectrum = avg_spectralon[col, :]
        avg_spectrum[avg_spectrum == 0] = np.nan  # Avoid division by zero
        
        # Normalize the column
        normalized_hypercube[:, col, :] = hypercube[:, col, :] / avg_spectrum
        
        # Replace infinities and NaNs with zeros for consistency
        normalized_hypercube[np.isinf(normalized_hypercube)] = 0
        normalized_hypercube[np.isnan(normalized_hypercube)] = 0
        
        
    base_filename = os.path.splitext(os.path.basename(HSIreader.dataset[idx]['data']))[0]
    save_path = os.path.join(save_folder, f"{base_filename}_ref.ref")
    spectral.envi.save_image(save_path, normalized_hypercube, dtype='float32', force=True)
      
    HSIreader.clear_cache()
    
    
