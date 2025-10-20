import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

def plot_targeted_cells_timeseries(toxin, trial_string, target_cells, 
                                   timeseries_dir, save_dir):
    """
    Plot calcium and voltage timeseries for targeted cells.
    
    Parameters:
    -----------
    toxin : str
        Name of the toxin for labeling
    trial_string : str
        Trial identifier string to find CSV files
    target_cells : list
        List of cell IDs to plot
    timeseries_dir : str or Path
        Directory containing timeseries CSV files
    save_dir : str or Path
        Directory to save output figures
    """
    
    # Convert paths to Path objects
    timeseries_dir = Path(timeseries_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files matching pattern
    voltage_files = list(timeseries_dir.glob(f'{trial_string}*voltage_transfected.csv'))
    ca_files = list(timeseries_dir.glob(f'{trial_string}*ca_transfected.csv'))
    
    if len(voltage_files) == 0:
        print(f"Error: No voltage CSV files found matching pattern '{trial_string}*voltage_transfected.csv'")
        return None
    if len(ca_files) == 0:
        print(f"Error: No calcium CSV files found matching pattern '{trial_string}*ca_transfected.csv'")
        return None
    
    # Use first matching file
    voltage_path = voltage_files[0]
    ca_path = ca_files[0]
    
    print(f"Loading calcium data from: {ca_path.name}")
    print(f"Loading voltage data from: {voltage_path.name}")
    
    # Load timeseries data - force numeric columns to float
    ca_df = pd.read_csv(ca_path, low_memory=False)
    voltage_df = pd.read_csv(voltage_path, low_memory=False)

    # Find cell_y column and convert all columns after it to numeric
    cell_y_idx = ca_df.columns.get_loc('cell_y')
    for col in ca_df.columns[cell_y_idx+1:]:
        ca_df[col] = pd.to_numeric(ca_df[col], errors='coerce')
    for col in voltage_df.columns[cell_y_idx+1:]:
        voltage_df[col] = pd.to_numeric(voltage_df[col], errors='coerce')
    
    # Get tab10 colors
    tab10_colors = plt.cm.tab10.colors
    ca_color = tab10_colors[1]  # Orange
    voltage_color = tab10_colors[0]  # Blue
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(8,3))
    #ax.set_facecolor('white')
    
    # Parameters
    dt = 0.2  # 0.2 second sampling rate (5 Hz)
    gaussian_sigma = 3.0
    y_spacing = 1.5  # Spacing between cells
    y_offset = 0
    
    # Store all traces for finding global min/max
    all_traces = []
    
    # Plot each target cell
    for cell_id in target_cells:
        # Find cell in dataframes
        ca_cell_data = ca_df[ca_df['cell_id'] == cell_id]
        voltage_cell_data = voltage_df[voltage_df['cell_id'] == cell_id]
        
        if len(ca_cell_data) == 0:
            print(f"Warning: Cell {cell_id} not found in calcium data")
            continue
        if len(voltage_cell_data) == 0:
            print(f"Warning: Cell {cell_id} not found in voltage data")
            continue
        
        # Find 'cell_y' column index and extract timeseries after it
        cell_y_idx = ca_df.columns.get_loc('cell_y')
        ca_raw_trace = ca_cell_data.iloc[0, cell_y_idx+1:].values.astype(float)
        voltage_raw_trace = voltage_cell_data.iloc[0, cell_y_idx+1:].values.astype(float)
        '''
        # DEBUG: Check what we're getting
        print(f"Cell {cell_id}:")
        print(f"  cell_y_idx: {cell_y_idx}")
        print(f"  Columns after cell_y: {ca_df.columns[cell_y_idx+1:].tolist()[:5]}")  
        print(f"  ca_raw_trace dtype: {ca_raw_trace.dtype}")
        print(f"  ca_raw_trace shape: {ca_raw_trace.shape}")
        print(f"  First few values: {ca_raw_trace[:5]}")
        '''
        # Calculate baseline (median)
        ca_baseline = np.median(ca_raw_trace)
        voltage_baseline = np.median(voltage_raw_trace)
        
        # Calculate ΔF/F
        ca_deltaff = (ca_raw_trace - ca_baseline) / ca_baseline
        voltage_deltaff = (voltage_raw_trace - voltage_baseline) / voltage_baseline
        
        # Center by subtracting median
        ca_centered = ca_deltaff - np.median(ca_deltaff)
        voltage_centered = voltage_deltaff - np.median(voltage_deltaff)
        
        # Apply Gaussian filter
        ca_filtered = ndimage.gaussian_filter1d(ca_centered, sigma=gaussian_sigma)
        voltage_filtered = ndimage.gaussian_filter1d(voltage_centered, sigma=gaussian_sigma)
        
        # Create time axis
        time_axis = np.arange(len(ca_filtered)) * dt
        
        # Plot calcium trace
        ax.plot(time_axis, ca_filtered + y_offset, 
                color=ca_color, linewidth=2, alpha=1.0,
                label='Ca²⁺ (ΔF/F)' if cell_id == target_cells[0] else '')
        
        # Plot voltage trace
        ax.plot(time_axis, voltage_filtered + y_offset, 
                color=voltage_color, linewidth=2, alpha=1.0,
                label='Voltage (-ΔF/F)' if cell_id == target_cells[0] else '')
        '''
        # Add cell ID label
        ax.text(-max(time_axis) * 0.02, y_offset, f'Cell {cell_id}',
                color='black', fontweight='bold', fontsize=12,
                ha='right', va='center')
        '''
        # Store traces for scale bar calculation
        all_traces.extend([ca_filtered + y_offset, voltage_filtered + y_offset])
        
        y_offset += y_spacing
    
    # Calculate positions for scale bars
    x_max = max(time_axis)
    y_min = min([min(trace) for trace in all_traces])
    y_max = max([max(trace) for trace in all_traces])
    y_range = y_max - y_min
    
    # Time scale bar (100s)
    time_scale_length = 100  # seconds
    time_scale_start = x_max - time_scale_length - 20
    scale_bar_y = y_min - y_range * 0.15
    
    ax.plot([time_scale_start, time_scale_start + time_scale_length], 
            [scale_bar_y, scale_bar_y], 'k-', linewidth=3)
    ax.text(time_scale_start + time_scale_length/2, scale_bar_y - y_range * 0.05, 
            '100 s', ha='center', va='top', fontsize=14)
    
    # Y scale bars (colored) - one bar per cell with separate Ca and Voltage labels
    scale_bar_height = 0.2  # 20% ΔF/F
    scale_bar_x_ca = x_max - 150  # Position calcium scale bar inside plot
    scale_bar_x_voltage = x_max - 100  # Position voltage scale bar inside plot

    # Draw scale bars for each cell at their respective y-offsets
    y_offset_temp = 0
    for cell_id in target_cells:
        # Calcium scale bar
        ax.plot([scale_bar_x_ca, scale_bar_x_ca], 
                [y_offset_temp, y_offset_temp + scale_bar_height], 
                color=ca_color, linewidth=6)
        
        # Voltage scale bar
        ax.plot([scale_bar_x_voltage, scale_bar_x_voltage], 
                [y_offset_temp, y_offset_temp + scale_bar_height], 
                color=voltage_color, linewidth=6)
        
        y_offset_temp += y_spacing

    # Add text labels below scale bars
    ax.text(scale_bar_x_ca, y_min - y_range * 0.08, 
            '20%', 
            ha='center', va='top', fontsize=12, 
            color=ca_color, fontweight='bold')
    
    ax.text(scale_bar_x_voltage, y_min - y_range * 0.08, 
            '20%', 
            ha='center', va='top', fontsize=12, 
            color=voltage_color, fontweight='bold')
    
    # Clean up axes
    ax.set_xlim(0, x_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    '''
    # Add title
    ax.set_title(f'{toxin} - {trial_string}', 
                fontsize=16, fontweight='bold', pad=20)
    '''
    # Add legend only for first cell
    ax.legend(loc='upper right', fontsize=12, frameon=True)
    
    plt.tight_layout()
    
    # Save figures
    png_path = save_dir / f'{toxin}_{trial_string}_tc.png'
    eps_path = save_dir / f'{toxin}_{trial_string}_tc.eps'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"PNG saved to: {png_path}")
    
    fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight', transparent=True)
    print(f"EPS saved to: {eps_path}")
    
    plt.show()
    return fig


# Example usage
if __name__ == "__main__":
    # Set parameters
    toxin = "4AP"  
    trial_string = "20250128_slip2_area4"  
    target_cells = [65,211,318 ]  
    
    # Set paths
    timeseries_dir = Path(r'R:\home\firefly_link\ca_voltage_imaging_working\results_1')
    save_dir = Path(r'R:\home\firefly_link\Calcium_Voltage_Imaging\MDA_MB_468\analysis\paper_figures\example_timeseries_toxin')
    
    # Plot
    fig = plot_targeted_cells_timeseries(
        toxin=toxin,
        trial_string=trial_string,
        target_cells=target_cells,
        timeseries_dir=timeseries_dir,
        save_dir=save_dir
    )