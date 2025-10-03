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
from PIL import Image
import tifffile
import pickle
import matplotlib

def lab2masks(seg):
    masks = []
    for i in range(1, seg.max() + 1):
        masks.append((seg == i).astype(int))
    return np.array(masks)

def draw_cell_circles(cell_coords, img_shape, radius=15):
    """Draw white circles at cell coordinates"""
    overlay = np.zeros((*img_shape, 3), dtype=np.uint8)
    
    for cell_id, (x, y) in cell_coords.items():
        # Create a circle mask
        Y, X = np.ogrid[:img_shape[0], :img_shape[1]]
        circle_mask = (X - x)**2 + (Y - y)**2 <= radius**2
        
        # Draw white circle outline (hollow)
        circle_outline = circle_mask.astype(int)
        inner_circle = (X - x)**2 + (Y - y)**2 <= (radius-2)**2
        circle_outline = circle_outline - inner_circle.astype(int)
        
        # Apply white color
        for c in range(3):
            overlay[:, :, c] = np.clip(overlay[:, :, c].astype(np.int32) + circle_outline * 255, 0, 255).astype(np.uint8)
    
    return np.clip(overlay, 0, 255).astype(np.uint8)

# Create cell outlines
def create_outline_overlay(masks_dict, img_shape, colors_dict):
    overlay = np.zeros((*img_shape, 3), dtype=np.uint8)
    struct = np.ones((3, 3))  # 2D structure for outline
    
    for cell_id, mask in masks_dict.items():
        # Create outline by dilating and subtracting original
        dilated = ndimage.binary_dilation(mask, structure=struct, iterations=3)
        outline = np.logical_xor(dilated, mask).astype(int)
        
        # Apply WHITE color instead of cell-specific colors
        for c in range(3):
            overlay[:, :, c] = np.clip(overlay[:, :, c].astype(np.int32) + outline * 255, 0, 255).astype(np.uint8)
            
    return overlay

def plot_ca_voltage_figure(voltage_img_path, ca_img_path, seg_path, ca_ts_path, voltage_ts_path, 
                          cell_ids=[276, 278], save_path=None):
    """
    Plot simultaneous calcium and voltage imaging data.
    
    Parameters:
    -----------
    voltage_img_path : str or Path
        Path to voltage TIFF image
    ca_img_path : str or Path
        Path to calcium TIFF image
    seg_path : str or Path
        Path to segmentation numpy file
    ca_ts_path : str or Path
        Path to calcium timeseries CSV file
    voltage_ts_path : str or Path
        Path to voltage timeseries CSV file
    cell_ids : list
        List of cell IDs to plot (default: [276, 278])
    save_path : str or Path
        Where to save the figure (optional)
    """
    
    # Load data
    print("Loading images...")
    voltage_img = tifffile.imread(voltage_img_path)
    ca_img = tifffile.imread(ca_img_path)
    #print(voltage_img.shape, ca_img.shape)
    # Enhance contrast for better visibility
    #voltage_img = np.clip(voltage_img * 1.5, 0, np.percentile(voltage_img, 99))
    #ca_img = np.clip(ca_img * 1.5, 0, np.percentile(ca_img, 99))
    
    print("Loading segmentation...")
    seg = np.load(seg_path, allow_pickle=True)
    
    print("Loading timeseries data...")
    ca_df = pd.read_csv(ca_ts_path)
    voltage_df = pd.read_csv(voltage_ts_path)
    
    
    # Extract timeseries for specified cells
    ca_data = {}
    voltage_data = {}

    for seg_cell_id in cell_ids:  # seg_cell_id will be 312, 314, etc.
        # Get the corresponding timeseries ID
        timeseries_id = seg_cell_id
        
        # Find the cell in the dataframes using the timeseries ID
        ca_cell_data = ca_df[ca_df['cell_id'] == timeseries_id]
        voltage_cell_data = voltage_df[voltage_df['cell_id'] == timeseries_id]
        
        if len(ca_cell_data) == 0:
            print(f"Warning: Cell {seg_cell_id} (timeseries ID {timeseries_id}) not found in calcium data")
            continue
        if len(voltage_cell_data) == 0:
            print(f"Warning: Cell {seg_cell_id} (timeseries ID {timeseries_id}) not found in voltage data")
            continue
            
        # Extract timeseries (excluding last 3 columns: cell_id, cell_x, cell_y)
        # Store using segmentation cell ID as key
        ca_data[seg_cell_id] = ca_cell_data.iloc[0, :-3].values
        voltage_data[seg_cell_id] = voltage_cell_data.iloc[0, :-3].values
        
    # Load masks from cellpose _seg.npy file
    #seg_data = seg.item()
    #seg_data = seg
    #masks_array = seg_data['masks']
    masks_array = seg
    
    # Check dimensions and resize if needed
    if masks_array.shape != voltage_img.shape:
        from scipy.ndimage import zoom
        scale_y = voltage_img.shape[0] / masks_array.shape[0]
        scale_x = voltage_img.shape[1] / masks_array.shape[1]
        masks_array = zoom(masks_array, (scale_y, scale_x), order=0)

    # Add debugging first
    print(f"Unique cell IDs in original segmentation: {np.unique(masks_array)[:20]}")
    print(f"Max cell ID: {np.max(masks_array)}")
    print(f"Total unique cells: {len(np.unique(masks_array)) - 1}")  # -1 for background

    # Check if our target cells exist in the segmentation
    for cell_id in cell_ids:
        cell_exists = cell_id in np.unique(masks_array)
        print(f"Cell {cell_id} exists in segmentation: {cell_exists}")
        if cell_exists:
            original_mask = (masks_array == cell_id).astype(int)
            print(f"Cell {cell_id} has {np.sum(original_mask)} pixels")

    # Extract masks directly for target cells (better approach)
    masks = {}
    for cell_id in cell_ids:
        if cell_id in ca_data:
            # Extract mask directly for this specific cell ID
            mask = (masks_array == cell_id).astype(int)
            if np.sum(mask) > 0:
                masks[cell_id] = mask
                print(f"Cell {cell_id}: extracted mask with {np.sum(mask)} pixels")
            else:
                print(f"Cell {cell_id}: no pixels found in segmentation")
    '''
    # Get cell coordinates from timeseries data
    cell_coords = {}
    for cell_id in cell_ids:
        if cell_id in ca_data:  # Only get coords for cells we have data for
            ca_cell_data = ca_df[ca_df['cell_id'] == cell_id]
            if len(ca_cell_data) > 0:
                cell_x = ca_cell_data['cell_y'].iloc[0]
                cell_y = ca_cell_data['cell_x'].iloc[0]
                cell_coords[cell_id] = (cell_x, cell_y)
                print(f"Cell {cell_id}: coordinates ({cell_x}, {cell_y})")
    
    # Create colors for cells
    colors_dict = {}
    cmap = matplotlib.cm.tab10
    for i, cell_id in enumerate(cell_coords.keys()):
        colors_dict[cell_id] = cmap(i / len(cell_coords))
    '''
    colors_dict = {}
    cmap = matplotlib.cm.tab10
    for i, cell_id in enumerate(masks.keys()):
        colors_dict[cell_id] = cmap(i / len(masks))

    # Create figure with specified layout
    fig = plt.figure(figsize=(12,8), constrained_layout=True)
    
    # Create grid: top 2/3, bottom 1/3
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.5], hspace=0.1, wspace=0.1)
    
    # Top left: Voltage image with cell outlines
    ax1 = fig.add_subplot(gs[0, 0])
    # For voltage image:
    '''
    circle_overlay = draw_cell_circles(cell_coords, voltage_img.shape, radius=15)
    ax1.imshow(voltage_img, cmap='gray', alpha=0.8)
    ax1.imshow(circle_overlay, alpha=0.8)
    '''
    voltage_overlay = create_outline_overlay(masks, voltage_img.shape, {})
    ax1.imshow(voltage_img, cmap='gray', alpha=0.8)
    ax1.imshow(voltage_overlay, alpha=0.8)
    ax1.set_title('Voltage Channel', fontsize=14)
    ax1.axis('off')
    scale_length = int(100 / 1.04)  # 100 μm scale bar
    plot_scalebar(ax1, voltage_img.shape[1] - scale_length - 10, 
                voltage_img.shape[0] - 20, scale_length, 5, color='white')
    
    
    # Top right: Calcium image with cell outlines
    ax2 = fig.add_subplot(gs[0, 1])
    '''
    circle_overlay = draw_cell_circles(cell_coords, ca_img.shape, radius=15)
    ax2.imshow(ca_img, cmap='gray', alpha=0.8)
    ax2.imshow(circle_overlay, alpha=0.8)
    '''
    ca_overlay = create_outline_overlay(masks, ca_img.shape, {})
    ax2.imshow(ca_img, cmap='gray', alpha=0.8)
    ax2.imshow(ca_overlay, alpha=0.8)
    ax2.set_title('Calcium Channel', fontsize=14)
    ax2.axis('off')
    plot_scalebar(ax2, ca_img.shape[1] - scale_length - 10, 
              ca_img.shape[0] - 20, scale_length, 5, color='white')
    
    '''
    # Update cell labels to use coordinates:
    for i, (cell_id, (x, y)) in enumerate(cell_coords.items()):
        ax1.text(x, y, str(i+1), color='white', fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        ax2.text(x, y, str(i+1), color='white', fontsize=12, fontweight='bold',
                ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    '''
    # Bottom: Timeseries plots
    ax3 = fig.add_subplot(gs[1, :])
    
    # Time axis (assuming same length for both ca and voltage)
    if ca_data:
        time_points = np.arange(len(list(ca_data.values())[0]))
        dt = 0.1  # Adjust this based on your actual sampling rate
        time_axis = time_points * dt
    
    # Plot timeseries for each cell
    y_offset = 0
    y_spacing = 2.0
    
    for i, cell_id in enumerate(sorted(ca_data.keys())):
        color = colors_dict[cell_id]
        
        # Plot calcium trace
        ca_trace = ca_data[cell_id]
        ca_normalized = (ca_trace - np.mean(ca_trace)) / np.std(ca_trace)
        ax3.plot(time_axis, ca_normalized + y_offset, 
                color=color, linewidth=3, label=f'Cell {cell_id} - Ca²⁺')
        
        # Plot voltage trace (offset slightly)
        voltage_trace = voltage_data[cell_id]
        voltage_normalized = (voltage_trace - np.mean(voltage_trace)) / np.std(voltage_trace)
        ax3.plot(time_axis, voltage_normalized + y_offset, 
                color=color, linewidth=3,
                label=f'Cell {cell_id} - Voltage')
        
        # Add cell ID label
        # Add cell ID label (fix the positioning)
        ax3.text(-max(time_axis) * 0.02, y_offset + 0.25, f'Cell {cell_id}',  # Changed from -0.05 to -0.02
                color=color, fontweight='bold', fontsize=11,
                ha='right', va='center')
        
        y_offset += y_spacing
    
    # Add scale bars to timeseries
    # X scale bar (100s = 500 timepoints at 5Hz)
    x_scale_length = 500  # timepoints
    x_scale_time = x_scale_length * dt  # convert to time
    y_pos = min([y_offset - y_spacing for y_offset in range(len(ca_data))]) - 1

    # X scale bar
    ax3.plot([max(time_axis) - x_scale_time, max(time_axis)], 
            [y_pos, y_pos], 'k-', linewidth=3)
    ax3.text(max(time_axis) - x_scale_time/2, y_pos - 0.3, '100 s', 
            ha='center', va='top', fontsize=10)

    # Y scale bar (2%)
    y_scale = 2.0  # 2% scale
    ax3.plot([max(time_axis) + 0.02*max(time_axis), max(time_axis) + 0.02*max(time_axis)], 
            [y_pos, y_pos + y_scale], 'k-', linewidth=3)
    ax3.text(max(time_axis) + 0.04*max(time_axis), y_pos + y_scale/2, '2%', 
            ha='left', va='center', fontsize=10, rotation=90)
    
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    # Add legend
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    #plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight', transparent=True)
        # Save EPS version
        eps_path = save_path.with_suffix('.eps')
        fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi = 300, transparent = True)
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig

def plot_scalebar(ax, x, y, length, height, thickness=2, color='white'):
    """Simple scale bar plotting function"""
    ax.add_patch(plt.Rectangle((x, y), length, thickness, 
                              facecolor=color, edgecolor=None))

def plot_segment_timeseries(ca_ts_path, voltage_ts_path, ca_raw_path, voltage_raw_path,
                           cell_id, start_time=420, end_time=500, save_path=None,
                           ca_scale_bar=0.02, voltage_scale_bar=0.15, gaussian_sigma=3.0):
    """
    Plot a segment of calcium and voltage timeseries for a specific cell with raw and filtered traces.
    
    Parameters:
    -----------
    ca_ts_path : str or Path
        Path to processed calcium timeseries CSV
    voltage_ts_path : str or Path  
        Path to processed voltage timeseries CSV
    ca_raw_path : str or Path
        Path to raw calcium timeseries CSV
    voltage_raw_path : str or Path
        Path to raw voltage timeseries CSV
    cell_id : int
        Timeseries cell ID (e.g., 278)
    start_time : float
        Start time in seconds (default: 420)
    end_time : float
        End time in seconds (default: 500)
    save_path : str or Path
        Base path for saving (will save both PNG and EPS)
    ca_scale_bar : float
        Scale bar size for calcium data (default: 0.02)
    voltage_scale_bar : float
        Scale bar size for voltage data (default: 0.15)
    gaussian_sigma : float
        Sigma for Gaussian filtering (default: 3.0)
    """
    
    from scipy import ndimage
    
    print(f"Loading timeseries for cell {cell_id}...")
    
    # Load timeseries data
    ca_df = pd.read_csv(ca_ts_path)
    voltage_df = pd.read_csv(voltage_ts_path)
    ca_raw_df = pd.read_csv(ca_raw_path)
    voltage_raw_df = pd.read_csv(voltage_raw_path)
    
    # Find the specific cell
    ca_cell_data = ca_df[ca_df['cell_id'] == cell_id]
    voltage_cell_data = voltage_df[voltage_df['cell_id'] == cell_id]
    ca_raw_cell_data = ca_raw_df[ca_raw_df['cell_id'] == cell_id]
    voltage_raw_cell_data = voltage_raw_df[voltage_raw_df['cell_id'] == cell_id]
    
    if len(ca_raw_cell_data) == 0 or len(voltage_raw_cell_data) == 0:
        print(f"Error: Cell {cell_id} not found in timeseries data")
        return None
    
    # Extract timeseries (excluding last 3 columns)
    ca_raw_trace = ca_raw_cell_data.iloc[0, :-3].values
    voltage_raw_trace = voltage_raw_cell_data.iloc[0, :-3].values
    
    # Calculate ΔF/F for raw data
    ca_baseline = np.median(ca_raw_trace)
    voltage_baseline = np.median(voltage_raw_trace)
    ca_raw_deltaff = (ca_raw_trace - ca_baseline) / ca_baseline
    voltage_raw_deltaff = (voltage_raw_trace - voltage_baseline) / voltage_baseline
    
    print(f"Calcium baseline (median F₀): {ca_baseline:.2f}")
    print(f"Voltage baseline (median F₀): {voltage_baseline:.2f}")
    
    # Create time axis (5 Hz sampling rate)
    dt = 0.2  # 0.2 second sampling rate (5 Hz = 1/5 = 0.2s per sample)
    time_axis = np.arange(len(ca_raw_trace)) * dt
    
    # Find indices for the time segment
    start_idx = int(start_time / dt)
    end_idx = int(end_time / dt)
    
    # Extract segment
    time_segment = time_axis[start_idx:end_idx]
    ca_segment = ca_raw_deltaff[start_idx:end_idx]
    voltage_segment = voltage_raw_deltaff[start_idx:end_idx]
    
    # Center the data by subtracting median (for overlaying)
    ca_centered = ca_segment - np.median(ca_segment)
    voltage_centered = voltage_segment - np.median(voltage_segment)
    
    # Apply Gaussian filtering
    ca_filtered = ndimage.gaussian_filter1d(ca_centered, sigma=gaussian_sigma)
    voltage_filtered = ndimage.gaussian_filter1d(voltage_centered, sigma=gaussian_sigma)
    
    # Get tab10 colors (orange and blue)
    tab10_colors = plt.cm.tab10.colors
    ca_color = tab10_colors[1]  # Orange
    voltage_color = tab10_colors[0]  # Blue
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    ax.set_facecolor('white')
    
    # Plot calcium traces
    # Raw data with transparency
    ax.plot(time_segment, ca_centered, 
            color=ca_color, linewidth=1, alpha=0.7, 
            label=r'Ca²⁺ Raw ($\Delta$F/F)')
    
    # Filtered data (opaque)
    ax.plot(time_segment, ca_filtered, 
            color=ca_color, linewidth=2, alpha=1.0,
            label=r'Ca²⁺ Gaussian Filtered ($\Delta$F/F)')
    
    # Plot voltage traces
    # Raw data with transparency
    ax.plot(time_segment, voltage_centered, 
            color=voltage_color, linewidth=1, alpha=0.7,
            label=r'Voltage Raw (-$\Delta$F/F)')
    
    # Filtered data (opaque)
    ax.plot(time_segment, voltage_filtered, 
            color=voltage_color, linewidth=2, alpha=1.0,
            label=r'Voltage Gaussian Filtered (-$\Delta$F/F)')
    
    # Add scale bars
    # Calculate positions
    y_min = min(min(ca_centered), min(voltage_centered))
    y_max = max(max(ca_centered), max(voltage_centered))
    y_range = y_max - y_min
    
    # X scale bar (10s)
    x_scale_length = 10  # seconds
    x_scale_start = max(time_segment) - x_scale_length - 5
    y_scale_pos = y_min - y_range * 0.15
    
    ax.plot([x_scale_start, x_scale_start + x_scale_length], 
            [y_scale_pos, y_scale_pos], 'k-', linewidth=3)
    ax.text(x_scale_start + x_scale_length/2, y_scale_pos - y_range * 0.08, '10 s', 
            ha='center', va='top', fontsize=14)
    
    # Calcium Y scale bar (colored)
    ca_scale_x = max(time_segment) - 15
    ca_scale_y_start = y_min - y_range * 0.05
    ax.plot([ca_scale_x, ca_scale_x], 
            [ca_scale_y_start, ca_scale_y_start + ca_scale_bar], 
            color=ca_color, linewidth=6)
    ax.text(ca_scale_x - 1, ca_scale_y_start + ca_scale_bar/2, 
            f'{ca_scale_bar*100:.0f}% $\Delta$F/F', 
            ha='right', va='center', fontsize=14, 
            color=ca_color, fontweight='bold')
    
    # Voltage Y scale bar (colored)
    voltage_scale_x = max(time_segment) - 5
    voltage_scale_y_start = y_min - y_range * 0.05
    ax.plot([voltage_scale_x, voltage_scale_x], 
            [voltage_scale_y_start, voltage_scale_y_start + voltage_scale_bar], 
            color=voltage_color, linewidth=6)
    ax.text(voltage_scale_x - 1, voltage_scale_y_start + voltage_scale_bar/2, 
            f'{voltage_scale_bar*100:.0f}% $\Delta$F/F', 
            ha='right', va='center', fontsize=14, 
            color=voltage_color, fontweight='bold')
    
    # Clean up axes
    ax.set_xlim(start_time, end_time)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, frameon=True)
    
    plt.tight_layout()
    
    # Save both formats if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PNG
        png_path = save_path.with_suffix('.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"PNG saved to: {png_path}")
        
        # Save EPS
        eps_path = save_path.with_suffix('.eps')
        fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=300, facecolor='white')
        print(f"EPS saved to: {eps_path}")
    
    plt.show()
    return fig

def extract_and_save_frames(tif_folder_path, save_dir, ca_frame_idx=None, voltage_frame_idx=None):
    """
    Extract specific calcium and voltage frames from interleaved TIF stack.
    Uses deinterleaving approach: even frames = voltage, odd frames = calcium.
    
    Parameters:
    -----------
    tif_folder_path : str or Path
        Path to folder containing TIF files
    save_dir : str or Path
        Directory to save the extracted frames
    ca_frame_idx : int, optional
        Frame index to extract for calcium channel (0-based indexing)
    voltage_frame_idx : int, optional
        Frame index to extract for voltage channel (0-based indexing)
    """
    
    if ca_frame_idx is None or voltage_frame_idx is None:
        print("Error: Both ca_frame_idx and voltage_frame_idx must be specified")
        return None
    
    tif_folder = Path(tif_folder_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and sort .tif files in the correct order (same as your code)
    tif_files = sorted([f for f in tif_folder.iterdir() if f.suffix == ".tif" and "ome" in f.name])
    
    # Custom sorting logic (same as your code)
    def custom_sort(filename):
        filename_str = str(filename)
        if filename_str.endswith("_Default.ome.tif"):
            return 0  # Highest priority
        elif filename_str.endswith("_Default_1.ome.tif"):
            return 1  # Second priority
        elif filename_str.endswith("_Default_2.ome.tif"):
            return 2  # Third priority
        return 3  # Any other files

    tif_files.sort(key=custom_sort)
    
    if not tif_files:
        print(f"No .ome.tif files found in {tif_folder}")
        return None
    
    print(f"Found TIF files: {[f.name for f in tif_files]}")
    
    # Load the first (main) TIF file (same as your approach)
    image_stack = tifffile.imread(str(tif_files[0]))
    
    if image_stack.ndim != 3:
        print(f"Error: Expected 3D image stack, got {image_stack.ndim}D")
        return None
    
    print(f"Loaded image stack shape: {image_stack.shape}")
    
    # Deinterleave: Split into 2 channels (same as your code)
    voltage_channel = image_stack[::2, :, :]  # Take every even frame (0, 2, 4, ...)
    calcium_channel = image_stack[1::2, :, :] # Take every odd frame (1, 3, 5, ...)
    
    print(f"Voltage channel shape: {voltage_channel.shape}")
    print(f"Calcium channel shape: {calcium_channel.shape}")
    
    # Check if requested frame indices are valid
    if ca_frame_idx >= calcium_channel.shape[0] or ca_frame_idx < 0:
        print(f"Error: Calcium frame index {ca_frame_idx} is out of range (0-{calcium_channel.shape[0]-1})")
        return None
    
    if voltage_frame_idx >= voltage_channel.shape[0] or voltage_frame_idx < 0:
        print(f"Error: Voltage frame index {voltage_frame_idx} is out of range (0-{voltage_channel.shape[0]-1})")
        return None
    
    # Extract the requested frames
    ca_frame = calcium_channel[ca_frame_idx]
    voltage_frame = voltage_channel[voltage_frame_idx]
    
    # Save the frames
    ca_save_path = save_dir / 'ca_frame.tif'
    voltage_save_path = save_dir / 'voltage_frame.tif'
    
    tifffile.imwrite(ca_save_path, ca_frame)
    tifffile.imwrite(voltage_save_path, voltage_frame)
    
    print(f"Calcium frame (index {ca_frame_idx}) saved to: {ca_save_path}")
    print(f"Voltage frame (index {voltage_frame_idx}) saved to: {voltage_save_path}")
    print(f"Calcium frame shape: {ca_frame.shape}")
    print(f"Voltage frame shape: {voltage_frame.shape}")
    
    return ca_frame, voltage_frame

def plot_raw_timeseries(ca_ts_path, voltage_ts_path, ca_raw_path, voltage_raw_path, 
                        cell_id, save_path=None, ca_scale_bar=0.1, voltage_scale_bar=0.05,
                        gaussian_sigma=3.0):
    """
    Plot full raw and filtered timeseries for calcium and voltage data.
    
    Parameters:
    -----------
    ca_ts_path : str or Path
        Path to processed calcium timeseries CSV
    voltage_ts_path : str or Path  
        Path to processed voltage timeseries CSV
    ca_raw_path : str or Path
        Path to raw calcium timeseries CSV
    voltage_raw_path : str or Path
        Path to raw voltage timeseries CSV
    cell_id : int
        Timeseries cell ID (e.g., 278)
    save_path : str or Path
        Base path for saving (will save both PNG and EPS)
    ca_scale_bar : float
        Scale bar size for calcium data (default: 0.1)
    voltage_scale_bar : float
        Scale bar size for voltage data (default: 0.05)
    gaussian_sigma : float
        Sigma for Gaussian filtering (default: 2.0)
    """
    
    print(f"Loading timeseries for cell {cell_id}...")
    
    # Load all timeseries data
    ca_df = pd.read_csv(ca_ts_path)
    voltage_df = pd.read_csv(voltage_ts_path)
    ca_raw_df = pd.read_csv(ca_raw_path)
    voltage_raw_df = pd.read_csv(voltage_raw_path)
    
    # Find the specific cell in all datasets
    ca_cell_data = ca_df[ca_df['cell_id'] == cell_id]
    voltage_cell_data = voltage_df[voltage_df['cell_id'] == cell_id]
    ca_raw_cell_data = ca_raw_df[ca_raw_df['cell_id'] == cell_id]
    voltage_raw_cell_data = voltage_raw_df[voltage_raw_df['cell_id'] == cell_id]
    
    # Check if cell exists in all datasets
    if len(ca_cell_data) == 0:
        print(f"Error: Cell {cell_id} not found in processed calcium data")
        return None
    if len(voltage_cell_data) == 0:
        print(f"Error: Cell {cell_id} not found in processed voltage data")
        return None
    if len(ca_raw_cell_data) == 0:
        print(f"Error: Cell {cell_id} not found in raw calcium data")
        return None
    if len(voltage_raw_cell_data) == 0:
        print(f"Error: Cell {cell_id} not found in raw voltage data")
        return None
    
    # Extract timeseries (excluding last 3 columns)
    ca_trace = ca_cell_data.iloc[0, :-3].values
    voltage_trace = voltage_cell_data.iloc[0, :-3].values
    ca_raw_trace = ca_raw_cell_data.iloc[0, :-3].values
    voltage_raw_trace = voltage_raw_cell_data.iloc[0, :-3].values
    
    # Calculate ΔF/F for raw data
    # Use median as baseline (F₀)
    ca_baseline = np.median(ca_raw_trace)
    voltage_baseline = np.median(voltage_raw_trace)
    
    # Calculate ΔF/F = (F - F₀) / F₀
    ca_raw_deltaff = (ca_raw_trace - ca_baseline) / ca_baseline
    voltage_raw_deltaff = (voltage_raw_trace - voltage_baseline) / voltage_baseline
    
    print(f"Calcium baseline (median F₀): {ca_baseline:.2f}")
    print(f"Voltage baseline (median F₀): {voltage_baseline:.2f}")
    
    # Create time axis (5 Hz sampling rate)
    dt = 0.2  # 0.2 second sampling rate (5 Hz)
    time_axis = np.arange(len(ca_raw_deltaff)) * dt
    
    # Center the ΔF/F data by subtracting median
    ca_raw_centered = ca_raw_deltaff - np.median(ca_raw_deltaff)
    voltage_raw_centered = voltage_raw_deltaff - np.median(voltage_raw_deltaff)
    
    # Apply Gaussian filtering to ΔF/F data for filtered traces
    ca_filtered = ndimage.gaussian_filter1d(ca_raw_centered, sigma=gaussian_sigma)
    voltage_filtered = ndimage.gaussian_filter1d(voltage_raw_centered, sigma=gaussian_sigma)
    
    # Get tab10 colors (blue and orange)
    tab10_colors = plt.cm.tab10.colors
    ca_color = tab10_colors[1]  # Orange (index 1)
    voltage_color = tab10_colors[0]  # Blue (index 0)
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(15, 6), facecolor='white')
    ax.set_facecolor('white')
    
    # Set up y-offset for overlapping traces
    y_offset = 0
    
    # Plot calcium traces
    # Raw data with transparency
    ax.plot(time_axis, ca_raw_centered + y_offset, 
            color=ca_color, linewidth=1, alpha=0.7, 
            label=f'Ca²⁺ Raw (ΔF/F)')
    
    # Filtered data (opaque)
    ax.plot(time_axis, ca_filtered + y_offset, 
            color=ca_color, linewidth=2, alpha=1.0,
            label=f'Ca²⁺ Gaussian Filtered (ΔF/F)')
    
    # Plot voltage traces
    # Raw data with transparency  
    ax.plot(time_axis, voltage_raw_centered + y_offset, 
            color=voltage_color, linewidth=1, alpha=0.7,
            label=f'Voltage Raw (-ΔF/F)')
    
    # Filtered data (opaque)
    ax.plot(time_axis, voltage_filtered + y_offset, 
            color=voltage_color, linewidth=2, alpha=1.0,
            label=f'Voltage Gaussian Filtered (-ΔF/F)')
    
    # Calculate positions for scale bars
    x_max = max(time_axis)
    y_min = min(min(ca_raw_centered), min(voltage_raw_centered)) - 0.1
    
    # Time scale bar (100s)
    time_scale_length = 100  # seconds
    time_scale_start = x_max - time_scale_length - 20
    
    ax.plot([time_scale_start, time_scale_start + time_scale_length], 
            [y_min, y_min], 'k-', linewidth=3)
    ax.text(time_scale_start + time_scale_length/2, y_min - 0.02, '100 s', 
            ha='center', va='top', fontsize=14)
    
    # Calculate better positions and make scale bars more visible
    y_data_range = max(max(ca_raw_centered), max(voltage_raw_centered)) - min(min(ca_raw_centered), min(voltage_raw_centered))
    
    # Calcium scale bar (colored) - positioned higher and thicker
    ca_scale_x = x_max - 200  # Position for calcium scale bar
    ca_scale_y_start = y_min + y_data_range * 0.1  # Start higher up
    ax.plot([ca_scale_x, ca_scale_x], 
            [ca_scale_y_start, ca_scale_y_start + ca_scale_bar], 
            color=ca_color, linewidth=6)  # Thicker line
    ax.text(ca_scale_x - 10, ca_scale_y_start + ca_scale_bar/2, 
        f'{ca_scale_bar*100:.0f}% $\Delta$F/F', ha='right', va='center', 
        fontsize=14, color=ca_color, fontweight='bold')
    
    # Voltage scale bar (colored) - positioned higher and thicker
    voltage_scale_x = x_max - 100  # Position for voltage scale bar
    voltage_scale_y_start = y_min + y_data_range * 0.1  # Start higher up
    ax.plot([voltage_scale_x, voltage_scale_x], 
            [voltage_scale_y_start, voltage_scale_y_start + voltage_scale_bar], 
            color=voltage_color, linewidth=6)  # Thicker line
    ax.text(voltage_scale_x - 10, voltage_scale_y_start + voltage_scale_bar/2, 
        f'{voltage_scale_bar*100:.0f}% $\Delta$F/F', ha='right', va='center', 
        fontsize=14, color=voltage_color, fontweight='bold')
    
    # Clean up axes
    ax.set_xlim(0, x_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add title
    ax.set_title(f'Cell {cell_id} - Raw and Filtered Timeseries (ΔF/F)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, frameon=True)
    
    plt.tight_layout()
    
    # Save both formats if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PNG with white background
        png_path = save_path.with_suffix('.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"PNG saved to: {png_path}")
        
        # Save EPS with white background
        eps_path = save_path.with_suffix('.eps')
        fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=300, facecolor='white')
        print(f"EPS saved to: {eps_path}")
    
    plt.show()
    return fig

# Set up paths 
home = Path.home()
if 'ys5320' in str(home):
    base_dir = Path(home, 'firefly_link/ca_voltage_imaging_working')
    data_dir = Path(home, 'firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')
else:
    base_dir = Path('R:/home/firefly_link/ca_voltage_imaging_working')
    data_dir = Path('R:/home/firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')
    
save_dir = Path(data_dir, 'paper_figures','example_tcs.png')
save_dir.parent.mkdir(parents=True, exist_ok=True)
# Image paths
image_dir = base_dir / 'single_frames'
image_dir = Path(r'C:\Users\Firefly\Desktop\single_frames')
voltage_img_path = image_dir / '20250226_slip5_area1_TRAM-34_1uM_1_MMStack_Default_voltage_aligned_frame.tif'
ca_img_path = image_dir / '20250226_slip5_area1_TRAM-34_1uM_1_MMStack_Default_ca_aligned_frame.tif'
 
# Segmentation path
seg_dir = base_dir / 'results_1'
seg_path = seg_dir / '20250226_slip5_area1_TRAM-34_1uM_masks_renumbered.npy'
 
# Timeseries paths
timeseries_dir = Path(data_dir, 'results_pipeline/20250226_slip5_area1')
ca_ts_path = timeseries_dir / 'pre_calcium_TRAM-34_1uM_20250226_slip5_area1.csv'
voltage_ts_path = timeseries_dir / 'pre_voltage_TRAM-34_1uM_20250226_slip5_area1.csv'
 
# Call the function
'''
fig = plot_ca_voltage_figure(voltage_img_path, ca_img_path, seg_path, ca_ts_path, voltage_ts_path,
                            cell_ids=[278], save_path= save_dir)

segment_save_path = Path(data_dir, 'paper_figures', 'eg_segment')
fig_segment = plot_segment_timeseries(ca_ts_path, voltage_ts_path, 
                                    cell_id=278, 
                                    start_time=480, end_time=580, 
                                    save_path=segment_save_path)

ca_frame, voltage_frame = extract_and_save_frames(
    tif_folder_path= Path(home, 'firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/20250226/slip5/area1/20250226_slip5_area1_TRAM-34_1uM_1'), 
    save_dir= Path(data_dir, 'paper_figures'),
    ca_frame_idx=2726,     # Specify calcium frame index
    voltage_frame_idx=1075 # Specify voltage frame index
)
'''
# Raw data paths
ca_raw_path = timeseries_dir / 'pre_calcium_TRAM-34_1uM_20250226_slip5_area1_raw.csv'
voltage_raw_path = timeseries_dir / 'pre_voltage_TRAM-34_1uM_20250226_slip5_area1_raw.csv'

# Save path for raw timeseries
raw_save_path = Path(data_dir, 'paper_figures', 'raw_timeseries_cell278')

# Call the raw timeseries function
fig_raw = plot_raw_timeseries(
    ca_ts_path=ca_ts_path,
    voltage_ts_path=voltage_ts_path, 
    ca_raw_path=ca_raw_path,
    voltage_raw_path=voltage_raw_path,
    cell_id=278,
    save_path=raw_save_path,
    ca_scale_bar=0.2,      # Adjust as needed
    voltage_scale_bar=0.2  # Adjust as needed
)

# Save path for segment
segment_save_path = Path(data_dir, 'paper_figures', 'eg_segment')

# Call the updated function
fig_segment = plot_segment_timeseries(
    ca_ts_path=ca_ts_path,
    voltage_ts_path=voltage_ts_path,
    ca_raw_path=ca_raw_path,
    voltage_raw_path=voltage_raw_path,
    cell_id=278, 
    start_time=480, 
    end_time=580, 
    save_path=segment_save_path,
    ca_scale_bar=0.2,      
    voltage_scale_bar=0.2,  
    gaussian_sigma=3.0       
)