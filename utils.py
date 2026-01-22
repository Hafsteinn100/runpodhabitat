import numpy as np
import pandas as pd
import base64
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data" 

def decode_patch(patch_str):
    """
    Decodes the base64 string sent by the judge into a numpy array.
    Used by api.py.
    """
    patch_bytes = base64.b64decode(patch_str)
    patch = np.frombuffer(patch_bytes, dtype=np.float32)
    return patch.reshape(15, 35, 35)

def extract_features(patch):
    """
    SPATIAL PYRAMID FEATURE EXTRACTION
    Extracts 330 features per image by analyzing the whole image 
    PLUS a 3x3 grid of sub-sections.
    """
    
    # --- Helper to calculate stats for a block of pixels ---
    def get_pixel_stats(pixels):
        # pixels shape: (15, N_pixels)
        mean = np.mean(pixels, axis=1) # Shape (15,)
        std = np.std(pixels, axis=1)   # Shape (15,)
        
        # Spectral Indices Stats (Mean ONLY)
        green = pixels[2]
        red = pixels[3]
        nir = pixels[7]
        epsilon = 1e-8
        
        ndvi_mean = np.mean((nir - red) / (nir + red + epsilon))
        ndwi_mean = np.mean((green - nir) / (green + nir + epsilon))
        gndvi_mean = np.mean((nir - green) / (nir + green + epsilon))
        
        return np.concatenate([
            mean, 
            std, 
            [ndvi_mean, ndwi_mean, gndvi_mean]
        ])

    # 1. Global Features (Whole Image)
    global_pixels = patch.reshape(15, -1)
    global_feats = get_pixel_stats(global_pixels)
    
    # 2. Grid Features (3x3 Split)
    # 35 pixels / 3 is approx 11. Slices: 0-11, 11-23, 23-35
    slices = [slice(0, 11), slice(11, 23), slice(23, 35)]
    
    grid_feats = []
    
    for h_s in slices:
        for w_s in slices:
            # Extract sub-patch
            sub_patch = patch[:, h_s, w_s]
            # Flatten and get stats
            sub_pixels = sub_patch.reshape(15, -1)
            grid_feats.append(get_pixel_stats(sub_pixels))
            
    # Concatenate everything: Global (33) + 9 * Grid (33) = 330 features
    return np.concatenate([global_feats] + grid_feats)

def load_data():
    """
    ULTIMATE LOADER: Includes 4x Data Augmentation.
    Rotates every image 0, 90, 180, and 270 degrees.
    """
    print("Loading raw data...")
    
    # Paths
    base_dir = DATA_DIR
    part1_path = base_dir / "train" / "patches_part1.npy"
    part2_path = base_dir / "train" / "patches_part2.npy"
    labels_path = base_dir / "train.csv"
    
    # Load Labels
    df = pd.read_csv(labels_path)
    original_labels = df["vistgerd_idx"].values
    
    # Load Patches
    if not part1_path.exists():
         # Fallback path check if files are not in a 'train' subfolder
         part1_path = base_dir / "patches_part1.npy"
         part2_path = base_dir / "patches_part2.npy"
         
    p1 = np.load(part1_path)
    p2 = np.load(part2_path)
    original_patches = np.concatenate([p1, p2], axis=0)
    
    print(f"Original dataset size: {original_patches.shape[0]} images")
    print("Applying 4x AUGMENTATION (Rotations 0, 90, 180, 270)...")
    
    X = []
    y = []
    
    # Augmentation Loop
    for patch, label in tqdm(zip(original_patches, original_labels), total=len(original_patches)):
        # 1. Original (0 deg)
        X.append(extract_features(patch))
        y.append(label)
        
        # 2. Rotate 90 deg
        p90 = np.rot90(patch, k=1, axes=(1, 2))
        X.append(extract_features(p90))
        y.append(label)
        
        # 3. Rotate 180 deg
        p180 = np.rot90(patch, k=2, axes=(1, 2))
        X.append(extract_features(p180))
        y.append(label)
        
        # 4. Rotate 270 deg
        p270 = np.rot90(patch, k=3, axes=(1, 2))
        X.append(extract_features(p270))
        y.append(label)
        
    X = np.array(X, dtype=np.float32) # Memory Optimization
    y = np.array(y)
    
    print(f"Final Augmented Dataset Shape: {X.shape}")
    return X, y