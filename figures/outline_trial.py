import sys
import numpy as np

# Compatibility fix for numpy._core
if not hasattr(np, '_core'):
    import types
    # Create the _core module
    _core_module = types.ModuleType('_core')
    _core_module.multiarray = np.core.multiarray
    _core_module.umath = np.core.umath
    
    # Add it to numpy
    np._core = _core_module
    
    # Add to sys.modules so pickle can find it
    sys.modules['numpy._core'] = _core_module
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
from pathlib import Path
from scipy import ndimage
import tifffile

def visualize_all_cell_ids(voltage_img_path, ca_img_path, seg_path, save_path=None):
    """
    Create overlay showing ALL cell IDs on both voltage and calcium images
    to help identify the correct cell numbers.
    """
    
    # Load data
    print("Loading images...")
    voltage_img = tifffile.imread(voltage_img_path)
    ca_img = tifffile.imread(ca_img_path)
    
    # Enhance contrast for better visibility
    voltage_img = np.clip(voltage_img * 2.0, 0, np.percentile(voltage_img, 99))
    ca_img = np.clip(ca_img * 2.0, 0, np.percentile(ca_img, 99))
    
    print("Loading segmentation...")
    seg = np.load(seg_path, allow_pickle=True)
    seg_data = seg.item()
    masks_array = seg_data['masks']
    
    # Check dimensions and resize if needed
    print(f"Original segmentation shape: {masks_array.shape}")
    print(f"Image shapes - Voltage: {voltage_img.shape}, Calcium: {ca_img.shape}")
    
    if masks_array.shape != voltage_img.shape:
        from scipy.ndimage import zoom
        scale_y = voltage_img.shape[0] / masks_array.shape[0]
        scale_x = voltage_img.shape[1] / masks_array.shape[1]
        print(f"Resizing segmentation with factors: y={scale_y:.3f}, x={scale_x:.3f}")
        masks_array = zoom(masks_array, (scale_y, scale_x), order=0)
    
    # Get all unique cell IDs
    unique_cells = np.unique(masks_array)
    unique_cells = unique_cells[unique_cells > 0]  # Remove background (0)
    print(f"Found {len(unique_cells)} cells with IDs: {unique_cells[:20]}...")  # Show first 20
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display voltage image
    ax1.imshow(voltage_img, cmap='gray')
    ax1.set_title('Voltage Channel with Cell IDs', fontsize=14)
    ax1.axis('off')
    
    # Display calcium image  
    ax2.imshow(ca_img, cmap='gray')
    ax2.set_title('Calcium Channel with Cell IDs', fontsize=14)
    ax2.axis('off')
    
    # Add cell ID labels for all cells
    print("Adding cell ID labels...")
    for cell_id in unique_cells:
        # Extract mask for this cell
        mask = (masks_array == cell_id).astype(int)
        
        if np.sum(mask) > 10:  # Only label cells with reasonable size (>10 pixels)
            # Find center of mass
            center = ndimage.center_of_mass(mask)
            
            if not (np.isnan(center[0]) or np.isnan(center[1])):
                # Add label to voltage image
                ax1.text(center[1], center[0], str(cell_id), 
                        color='yellow', fontsize=8, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
                
                # Add label to calcium image
                ax2.text(center[1], center[0], str(cell_id), 
                        color='yellow', fontsize=8, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
                
                # Optional: Add colored outline for better visibility
                struct = np.ones((3, 3))
                dilated = ndimage.binary_dilation(mask, structure=struct, iterations=2)
                outline = np.logical_xor(dilated, mask).astype(int)
                
                if np.sum(outline) > 0:
                    # Create overlay
                    y_coords, x_coords = np.where(outline)
                    ax1.scatter(x_coords, y_coords, c='red', s=0.1, alpha=0.3)
                    ax2.scatter(x_coords, y_coords, c='red', s=0.1, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Cell ID visualization saved to: {save_path}")
    
    plt.show()
    
    # Print summary info
    print(f"\nSummary:")
    print(f"Total cells found: {len(unique_cells)}")
    print(f"Cell ID range: {unique_cells.min()} to {unique_cells.max()}")
    print(f"Looking for cells 276 and 278:")
    print(f"  Cell 276 exists: {276 in unique_cells}")
    print(f"  Cell 278 exists: {278 in unique_cells}")
    
    return fig

# Set up paths
home = Path.home()
if 'ys5320' in str(home):
    base_dir = Path(home, 'firefly_link/ca_voltage_imaging_working')
    data_dir = Path(home, 'firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')
else:
    base_dir = Path('R:/home/firefly_link/ca_voltage_imaging_working')
    data_dir = Path('R:/home/firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')

# Image paths
image_dir = Path(r'C:\Users\Firefly\Desktop\single_frames')
voltage_img_path = image_dir / '20250226_slip5_area1_TRAM-34_1uM_1_MMStack_Default_voltage_aligned_frame.tif'
ca_img_path = image_dir / '20250226_slip5_area1_TRAM-34_1uM_1_MMStack_Default_ca_aligned_frame.tif'

# Segmentation path
seg_dir = base_dir / 'MDA_MB_468_brightfield'
seg_path = seg_dir / '20250226_slip5_area1_brightfield_before_MMStack_Default.ome_seg.npy'

# Save path
save_path = Path(data_dir, 'paper_figures', 'all_cell_ids_visualization.png')

# Run the visualization
fig = visualize_all_cell_ids(voltage_img_path, ca_img_path, seg_path, save_path)