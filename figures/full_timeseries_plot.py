import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
home = Path.home()
if 'ys5320' in str(home):
    base_dir = Path(home, 'firefly_link/ca_voltage_imaging_working')
    data_dir = Path(home, 'firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')
else:
    base_dir = Path('R:/home/firefly_link/ca_voltage_imaging_working')
    data_dir = Path('R:/home/firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')

results_pipeline_dir = data_dir / 'results_pipeline'
dataframes_dir = data_dir / 'dataframes'
df_path = dataframes_dir / 'MDA_MB_468_dataframe_tc_extracted.csv'

def plot_timeseries_for_trial(toxin, trial_string, data_type='voltage'):
    """
    Plot concatenated pre and post timeseries (horizontally) for each cell
    Each row shows one cell's pre+post timeseries
    """
    fig, ax = plt.subplots(1, 1, figsize=(10,20))
    
    # Load both segments
    segment_data = {}
    segment_files = {}
    
    for segment in ['pre', 'post']:
        trial_dir = results_pipeline_dir / trial_string
        pattern = f"{segment}_{data_type}*{trial_string}_raw.csv"
        matching_files = list(trial_dir.glob(pattern))
        
        if matching_files:
            segment_files[segment] = matching_files[0]
            data = pd.read_csv(matching_files[0])
            if 'cell_id' in data.columns:
                timeseries = data.drop(['cell_id', 'cell_x', 'cell_y'], axis=1)
            else:
                timeseries = data.iloc[:, :-3]
            segment_data[segment] = timeseries
        else:
            segment_data[segment] = None
            print(f"No file found for {segment}")
    
    # Check if both segments exist
    if segment_data['pre'] is None or segment_data['post'] is None:
        ax.text(0.5, 0.5, 'Missing pre or post data', 
               ha='center', va='center', transform=ax.transAxes)
        plt.close()
        return
    
    # Get dimensions
    n_cells = min(len(segment_data['pre']), len(segment_data['post']))
    pre_frames = segment_data['pre'].shape[1]
    post_frames = segment_data['post'].shape[1]
    total_frames = pre_frames + post_frames
    
    # Create time axis
    pre_time = np.arange(pre_frames) / 5.0
    post_time = (np.arange(post_frames) + pre_frames) / 5.0
    
    # Calculate channel spacing
    all_ranges = []
    for i in range(n_cells):
        pre_range = np.ptp(segment_data['pre'].iloc[i].values)
        post_range = np.ptp(segment_data['post'].iloc[i].values)
        all_ranges.extend([pre_range, post_range])
    
    max_range = max(all_ranges) if all_ranges else 1
    channel_spacing = max_range * 1.5
    
    # Plot each cell
    pre_stds = []
    post_stds = []
    
    for cell_idx in range(n_cells):
        offset = cell_idx * channel_spacing
        
        # Get traces
        pre_trace = segment_data['pre'].iloc[cell_idx].values
        post_trace = segment_data['post'].iloc[cell_idx].values
        
        # Calculate stds
        pre_std = np.std(pre_trace)
        post_std = np.std(post_trace)
        pre_stds.append(pre_std)
        post_stds.append(post_std)
        
        # Plot pre (blue)
        ax.plot(pre_time, pre_trace + offset, alpha=0.6, linewidth=0.8, color='blue')
        
        # Plot post (red)
        ax.plot(post_time, post_trace + offset, alpha=0.6, linewidth=0.8, color='red')
        
        # Add cell label on the left
        ax.text(-0.01 * np.max(post_time), offset, f'{cell_idx}', 
               verticalalignment='center', fontsize=8, fontweight='bold')
    
    # Add vertical line separating pre and post
    transition_time = pre_frames / 5.0
    ax.axvline(x=transition_time, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(transition_time, (n_cells-0.5) * channel_spacing, 'PRE | POST', 
           ha='center', va='bottom', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Cell (with offset)', fontsize=12)
    ax.set_title(f'{trial_string} - {data_type} - PRE + POST (n={n_cells} cells)', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, np.max(post_time))
    ax.set_ylim(-channel_spacing*0.5, (n_cells-0.5) * channel_spacing)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='PRE'),
                      Line2D([0], [0], color='red', lw=2, label='POST')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    '''
    # Print summary statistics
    print(f"\n{trial_string} - {data_type}:")
    print(f"  Number of cells: {n_cells}")
    print(f"  PRE duration: {pre_time[-1]:.1f} seconds")
    print(f"  POST duration: {post_time[-1] - transition_time:.1f} seconds")
    print(f"\nPRE:")
    print(f"  Mean std across cells: {np.mean(pre_stds):.4f}")
    print(f"  Median std across cells: {np.median(pre_stds):.4f}")
    print(f"  Std range: [{np.min(pre_stds):.4f}, {np.max(pre_stds):.4f}]")
    print(f"\nPOST:")
    print(f"  Mean std across cells: {np.mean(post_stds):.4f}")
    print(f"  Median std across cells: {np.median(post_stds):.4f}")
    print(f"  Std range: [{np.min(post_stds):.4f}, {np.max(post_stds):.4f}]")
    '''
    plt.tight_layout()
    
    # Save figure
    save_dir = data_dir / 'timeseries_plots'
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'{toxin}_{trial_string}_{data_type}_timeseries_concat.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    plt.show()

def plot_all_trials_for_toxin(toxin, data_type='voltage'):
    """
    Plot timeseries for all trials of a given toxin
    """
    # Load dataframe and filter
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n']
    toxin_trials = df[df['expt'].str.contains(toxin, na=False)]['trial_string'].unique()
    
    print(f"Found {len(toxin_trials)} trials for toxin '{toxin}': {list(toxin_trials)}")
    
    for trial in toxin_trials:
        print(f"\n{'='*60}")
        print(f"Plotting trial: {trial}")
        print('='*60)
        plot_timeseries_for_trial(toxin, trial, data_type)

if __name__ == "__main__":
    # Plot timeseries for 4AP
    plot_all_trials_for_toxin('Ca_free', data_type='voltage')
    
    # You can also plot a specific trial
    # plot_timeseries_for_trial('4AP', 'your_trial_string_here', data_type='voltage')
    
    # Or plot calcium data
    # plot_all_trials_for_toxin('4AP', data_type='calcium')