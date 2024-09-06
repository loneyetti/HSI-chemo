import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.measure import regionprops
import pickle

main_data_folder = "D:\\Test\\Bitou_1"  
dataset =HsiDataset(main_data_folder,data_ext='ref')

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

    nb_pca_comp =3
    pca = PCA(n_components=nb_pca_comp)
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(np.shape(pca_loadings)[1]):
        plt.figure()
        plt.plot(wv,pca_loadings[:,i],default_colors[i])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")  
        lab= 'PC'+str(i+1)
        plt.title(lab) 
        plt.grid()  
    plt.show(block=False)
    
    score_img = HSIreader.project_pca_scores(pca_loadings)
   
    # for s in range(pca_loadings.shape[1]):
    #     plt.figure()
    #     plt.imshow(score_img[:,:,s])
    #     plt.title(f'Score image PC{s+1}')
    #     plt.axis('off')
    #     plt.show(block=False)
       
    score_pc_ref = score_img[:,:,1]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=2)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    # plt.figure()
    # plt.imshow(segmented)
    # plt.show(block=False)
    
    labeled_image = label(segmented)
    
    binary_image = labeled_image > 0
    
    filled_binary_image = binary_fill_holes(binary_image)
    
    # plt.figure()
    # plt.imshow(filled_binary_image)
    # plt.show(block=False)
    
    labeled_image = label(filled_binary_image)
    labeled_image= label(remove_small_objects(labeled_image > 0, min_size=20))
    
    
    color_image = color_labels(labeled_image)
        
      # plt.figure()
      # plt.imshow(color_image)
      # plt.title('Color-Mapped Labeled Image')
      # plt.axis('off')
      # plt.show()
    
    
    regions = regionprops(labeled_image)
    print(f"Number of regions found: {len(regions)}")
    object_data = []
    
    for region in regions:
        # Get the object ID (label)
        obj_id = region.label
        
         # Skip the background (label 0)
        if obj_id == 0:
            continue
        
        # Get the centroid coordinates
        centroid = region.centroid
        
        # Get the coordinates of all pixels belonging to this object
        pixel_coords = np.array(region.coords)  # (num_pixels, 2) array
        
        # Store in dictionary
        object_data.append({
            'id': obj_id,
            'centroid': centroid,
            'pixels': pixel_coords
        })

    object_data.sort(key=lambda x: (x['centroid'][1], x['centroid'][0]))

# Convert list to dictionary with ordered keys
ordered_object_data = {obj['id']: obj for obj in object_data}

object_folder = os.path.join(main_data_folder, 'object')
object_file = 'object_data.pkl'

# Create the object folder if it doesn't exist
os.makedirs(object_folder, exist_ok=True)

# Full path for saving the file
file_path = os.path.join(object_folder, object_file)

# Save the ordered dictionary to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(ordered_object_data, f)