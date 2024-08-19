import os
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt

class HsiDataset:
    def __init__(self, main_dir, data_ext="dat", hdr_ext="hdr"):
        self.main_dir = main_dir
        self.data_ext = data_ext
        self.hdr_ext = hdr_ext
        
        self.data = []
        self.hdr = []

        self._collect_paths()
        
        
    def _collect_paths(self):
    # Walk through the directory and collect paths for raw data and header files
        for root, _, files in os.walk(self.main_dir):
            for file in files:
                if file.endswith(self.data_ext):
                    data_path = os.path.join(root, file)
                    hdr_path = self._get_corresponding_hdr(data_path)
                    
                    # Check if the corresponding header file exists
                    if os.path.exists(hdr_path):
                        self.data.append(data_path)
                        self.hdr.append(hdr_path)
                        
                    else:
                        # Skip the data file if no corresponding hdr file is found
                        print(f"Warning: No header file found for {data_path}, skipping this file.")
                        
    def _get_corresponding_hdr(self, data_path):
        # Get the header path by replacing the data file extension with the header extension
        base_path, _ = os.path.splitext(data_path)
        return f"{base_path}.{self.hdr_ext}"
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        # Allows indexing into the dataset to get a data-header pair
        if idx >= len(self.data):
            raise IndexError("Index out of range")
        return {"data": self.data[idx], "hdr": self.hdr[idx]}
   
    def get_all_paths(self):
        # Returns a list of all data-header pairs
        return [{"data": d, "hdr": h} for d, h in zip(self.data, self.hdr)]
    
    
    
class HsiReader:
    def __init__(self, dataset):
        if not isinstance(dataset, HsiDataset):
            raise TypeError("dataset must be an instance of HsiDataset")
        self.dataset = dataset
        self.current_image = None
        self.current_metadata = None
        self.current_idx = None
        self.hypercube = None  

        
    def read_image(self, idx):
        """
        Reads the image metadata for a specific index on-demand.
        Stores the metadata and file paths temporarily in memory.
        """
        if idx >= len(self.dataset):
            raise IndexError("Index out of range")

        data_path = self.dataset[idx]["data"]
        hdr_path = self.dataset[idx]["hdr"]

        try:
            # Read the image and metadata using the spectral library
            img = envi.open(hdr_path, data_path)
            metadata=img.metadata
            metadata={k.lower(): v for k, v in metadata.items()} 
            self.current_metadata = metadata  # Store the metadata
            self.current_image = img  # Store the image object for later loading
            self.current_idx = idx  # Track which image is currently read

        except Exception as e:
            print(f"Error reading {data_path} and {hdr_path}: {e}")
            self.current_metadata = None
            self.current_image = None
            self.current_idx = None
            
    def get_hsi(self, idx=None):
        """
        Loads the image data for the specified index if provided.
        If no index is provided, uses the current index, defaulting to not load anything if none is read.
        """
        if idx is not None:
            self.read_image(idx)

        # Ensure the hypercube is loaded
        if self.hypercube is None:
            if self.current_image is not None:
                try:
                    self.hypercube = self.current_image.load()
                except Exception as e:
                    print(f"Error loading image data: {e}")
                    return None
            else:
                print("No image has been read yet and no index provided. Doing nothing.")
                return None
        
        return self.hypercube
      
             
    def get_rgb(self, idx=None):
        if idx is not None:
            self.read_image(idx)
            
        if self.hypercube is None:
            self.get_hsi()
            
        if self.hypercube is None or self.current_metadata is None:
            print("No image or metadata available. Ensure `read_image` has been called.")
            return None
        
        try:
            rgb_channels = [int(i) for i in self.current_metadata.get('default bands', [])]
            print(f"RGB channels: {rgb_channels}")

            # Construct the RGB image
            rgb_image = np.asarray(self.hypercube[:, :, rgb_channels]).astype(np.float32)
            
            # Normalize the RGB image
            for ch in range(3):
                channel = rgb_image[:, :, ch]
                channel_min, channel_max = channel.min(), channel.max()
                if channel_max > channel_min:
                    rgb_image[:, :, ch] = (channel - channel_min) / (channel_max - channel_min) * 255
                else:
                    rgb_image[:, :, ch] = 0
            
            return rgb_image.astype(np.uint8)
        
        except KeyError:
            print("No 'default bands' key found in metadata.")
            return None
        
        
      
    def get_wavelength(self, idx=None):
        """
        Retrieves the wavelength information from the current image's metadata.
        If an index is provided, it will read the image at that index.
        """
        if idx is not None:
            self.read_image(idx)
        
        if self.current_metadata is None:
            print("No metadata available")
            return None
        
        wv = np.array([np.float32(i) for i in self.current_metadata.get('wavelength', [])])
        return wv    
         
    def clear_cache(self):
        """
        Clears the current image and metadata from memory.
        """
        self.current_image = None
        self.current_metadata = None
        self.current_idx = None
        self.hypercube = None
        
    def get_spectrum(self, idx=None):
        """
        Retrieves spectral information for the image at the specified index.
        If no index is provided, uses the current index.
        """
        if idx is not None:
            self.read_image(idx)
                  
        if self.hypercube is None:
            if self.current_image is None:
                print("No image or hypercube available. Ensure `read_image` has been called.")
                return None
            else:
                try:
                    self.hypercube = self.current_image.load()
                except Exception as e:
                    print(f"Error loading image data: {e}")
                    return None
                
        if self.current_metadata is None or self.hypercube is None:
            print("No image or metadata available.")
            return None

        try:
            rgb_image = self.get_rgb()
            if rgb_image is None:
                print("Failed to get RGB image.")
                return None
            wv =self.get_wavelength()

            
            all_spec = []
            all_pos = []
            color_list = plt.cm.tab20(np.linspace(0, 1, 20))
         
            legends = []
          
            fig2, ax2 = plt.subplots(constrained_layout=True)
            ax2.imshow(rgb_image)
            plt.title("Click on the image to sample spectra. Press Enter to finish.")
            plt.draw()
            
            pts = plt.ginput(-1, timeout=-1)
            
            fig, ax = plt.subplots()
            for idx, pos in enumerate(pts):
                x, y = int(pos[0]), int(pos[1])
                
                color = color_list[idx % len(color_list)]
                
                ax2.plot(x, y, 'x', color=color, markersize=8)
                ax2.text(x, y - 50, f'({x}, {y})', color=color, fontsize=10, verticalalignment='bottom', horizontalalignment='center')
                
                legend_marker = ax.scatter( [], [], marker='x', s=50, color=color, label=f'x={x}, y={y}')
                legends.append(legend_marker)
                
                spectrum = self.hypercube[y, x, :].reshape(-1)
                all_spec.append(spectrum)
                all_pos.append((x, y))
            
                ax.plot(wv, spectrum, label=f'({x}, {y})', color=color)
                ax.legend(handles=legends, loc='center left',
                  bbox_to_anchor=(1, 0.5), title='samples')
                
            ax.legend(handles=legends, loc='center left', bbox_to_anchor=(1, 0.5), title='Samples')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")
            plt.grid()
            plt.show()
            
         
            
            return all_spec, all_pos
        
        except Exception as e:
            print(f"Error processing spectrum: {e}")
            return None, None
            
            
