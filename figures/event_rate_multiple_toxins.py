import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from statannotations.Annotator import Annotator
from vsd_cancer.functions import stats_functions as statsf

# Set up paths based on environment
home = Path.home()
if 'ys5320' in str(home):
    base_dir = Path(home, 'firefly_link/ca_voltage_imaging_working')
    data_dir = Path(home, 'firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')
else:
    base_dir = Path('R:/home/firefly_link/ca_voltage_imaging_working')
    data_dir = Path('R:/home/firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')

# Define paths
results_pipeline_dir = data_dir / 'results_profiles'
dataframes_dir = data_dir / 'dataframes'
df_path = dataframes_dir / 'long_acqs_MDA_MB_468_all.csv'
#df_path = dataframes_dir / 'MDA_MB_468_dataframe_tc_extracted.csv'
#df_path = dataframes_dir / 'MDA_MB_468_segmented_results4.csv'
save_dir = data_dir / 'toxin_analysis_plots'
save_dir.mkdir(parents=True, exist_ok=True)

# Comprehensive toxin display labels with units
TOXIN_LABELS = {
    '4AP': '5 mM 4-AP',
    'Ca_free': r'Ca$^{2+}$ Free',
    'ATP': '1 mM ATP-$\gamma$S',
    'cbx': r'100 $\mu$M CBX',
    'heparin': '5 mg/mL Heparin',
    'TRAM-34_1uM_JEDIonly': r'1 $\mu$M TRAM-34',
    'TRAM-34': r'1 $\mu$M TRAM-34',
    'TTA-A2': r'10 $\mu$M TTA-A2',
    'Carbachol': r'100 $\mu$M Carbachol',
    'CBA': r'50 $\mu$M CBA',
    'BAY1797': r'10 $\mu$M BAY1797',
    'S-Bayk': r'10 $\mu$M S-Bayk',
    'Nifepidine': r'10 $\mu$M Nifedipine',
    'Ani9': r'10 $\mu$M Ani9',
    'dantrolene': r'10 $\mu$M Dantrolene',
    'Thapsigargin': r'1 $\mu$M Thapsigargin',
    'A01': r'50 $\mu$M A01',
    'PPADS': r'100 $\mu$M PPADS',
    'YM58483': r'10 $\mu$M YM58483',
    'L-15': 'L-15 Control',
    'DMSO_0.1%': '0.1% DMSO Control'
}

def load_timeseries_data_by_toxin(toxin, data_type='voltage', segment='pre', include_post_folders=False):
    """
    Load and concatenate timeseries data for a specific toxin, data type, and segment
    This gets ALL cells, including those without events
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n_focus']
    
    # Get ALL trials (paired + unpaired)
    toxin_df = df[df['expt'].str.contains(toxin, case=False, na=False)]
    
    print(f"\n=== Loading TIMESERIES data: {data_type} {segment} for toxin: {toxin} ===")
    if include_post_folders and segment == 'post':
        print(f"  (include_post_folders=True: will load from BOTH normal and _post folders)")
    
    all_data = []
    successful_trials = []
    
    # Group by trial_string to handle both regular and _post trials
    for trial_string in toxin_df['trial_string'].unique():
        trial_rows = toxin_df[toxin_df['trial_string'] == trial_string]
        
        # Determine which folders to check
        folders_to_check = [trial_string]  # Always check the normal folder first
        
        # If segment is 'post' and include_post_folders is True, ALSO check _post folder
        if segment == 'post' and include_post_folders:
            # Check if there's a row with expt ending in _post for this trial_string
            post_trials = trial_rows[trial_rows['expt'].str.endswith('_post', na=False)]
            if len(post_trials) > 0:
                # ALSO check the _post folder
                folders_to_check.append(f"{trial_string}_post")
                print(f"  → {trial_string}: Will load from BOTH normal and _post folders")
        
        # Try loading from each folder
        for folder_name in folders_to_check:
            trial_dir = results_pipeline_dir / folder_name
            
            # Use glob to find matching files
            # If using _post folder, the filename also has _post in the trial_string
            if folder_name.endswith('_post'):
                pattern = f"{segment}_{data_type}*{trial_string}_post_raw.csv"
            else:
                pattern = f"{segment}_{data_type}*{trial_string}_raw.csv"
            matching_files = list(trial_dir.glob(pattern))
            
            if matching_files:
                file_path = matching_files[0]
                try:
                    # Load the data
                    trial_data = pd.read_csv(file_path)
                    
                    # Remove position columns and keep only timeseries
                    if 'cell_id' in trial_data.columns:
                        timeseries_data = trial_data.drop(['cell_id', 'cell_x', 'cell_y'], axis=1)
                    else:
                        # Assume last 3 columns are position data
                        timeseries_data = trial_data.iloc[:, :-3]
                    
                    # Add trial identifier
                    # IMPORTANT: Keep track of which folder this came from
                    timeseries_data['trial_string'] = trial_string
                    timeseries_data['folder_source'] = folder_name  # Track source folder
                    timeseries_data['cell_index'] = range(len(timeseries_data))
                    
                    all_data.append(timeseries_data)
                    successful_trials.append(folder_name)
                    print(f"✓ {trial_string} (folder: {folder_name}): Successfully loaded ({len(timeseries_data)} cells)")
                    
                except Exception as e:
                    print(f"✗ {trial_string} (folder: {folder_name}): Error loading - {e}")
            else:
                if folder_name in folders_to_check:
                    print(f"✗ {trial_string} (folder: {folder_name}): File not found - {trial_dir / pattern}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Re-index cells globally after combining
        combined_data['global_cell_index'] = range(len(combined_data))
        return combined_data, successful_trials
    else:
        return None, []

def load_event_data_by_toxin(toxin, data_type='voltage', segment='pre', include_post_folders=False):
    """
    Load and concatenate event data from individual trial folders for a specific toxin
    """
    # Load metadata
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n_focus']
    
    # Get ALL trials (paired + unpaired)
    toxin_df = df[df['expt'].str.contains(toxin, case=False, na=False)]
    
    print(f"\n=== Loading EVENT data: {data_type} {segment} for toxin: {toxin} ===")
    if include_post_folders and segment == 'post':
        print(f"  (include_post_folders=True: will load from BOTH normal and _post folders)")
    
    all_events = []
    
    # Group by trial_string to handle both regular and _post trials
    for trial_string in toxin_df['trial_string'].unique():
        trial_rows = toxin_df[toxin_df['trial_string'] == trial_string]
        
        # Determine which folders to check
        folders_to_check = [trial_string]  # Always check the normal folder first
        
        # If segment is 'post' and include_post_folders is True, ALSO check _post folder
        if segment == 'post' and include_post_folders:
            # Check if there's a row with expt ending in _post for this trial_string
            post_trials = trial_rows[trial_rows['expt'].str.endswith('_post', na=False)]
            if len(post_trials) > 0:
                # ALSO check the _post folder
                folders_to_check.append(f"{trial_string}_post")
                print(f"  → {trial_string}: Will load from BOTH normal and _post folders")
        
        # Try loading from each folder
        for folder_name in folders_to_check:
            trial_dir = results_pipeline_dir / folder_name
            
            if not trial_dir.exists():
                print(f"✗ {trial_string} (folder: {folder_name}): Folder not found")
                continue
            
            # Try different possible event file patterns
            # If using _post folder, the filename also has _post in the trial_string
            if folder_name.endswith('_post'):
                possible_patterns = [
                    f"events_{data_type}_{segment}_{trial_string}_post_simple_QC_final.csv",
                    f"events_{data_type}_{segment}_{trial_string}_post_QC_final.csv",
                    f"events_{data_type}_{toxin}_{segment}_{trial_string}_post_simple_QC_final.csv",
                ]
            else:
                possible_patterns = [
                    f"events_{data_type}_{segment}_{trial_string}_simple_QC_final.csv",
                    f"events_{data_type}_{segment}_{trial_string}_QC_final.csv",
                    f"events_{data_type}_{toxin}_{segment}_{trial_string}_simple_QC_final.csv",
                ]
            
            event_file_found = False
            for pattern in possible_patterns:
                file_path = trial_dir / pattern
                if file_path.exists():
                    try:
                        event_data = pd.read_csv(file_path)
                        event_data = event_data.drop_duplicates()
                        
                        if 'trial_string' not in event_data.columns:
                            event_data['trial_string'] = trial_string
                        
                        # Track which folder this came from
                        event_data['folder_source'] = folder_name
                        
                        all_events.append(event_data)
                        event_file_found = True
                        print(f"✓ {trial_string} (folder: {folder_name}): Successfully loaded ({len(event_data)} events)")
                        break
                    except Exception as e:
                        print(f"✗ {trial_string} (folder: {folder_name}): Error loading - {e}")
            
            if not event_file_found:
                print(f"✗ {trial_string} (folder: {folder_name}): No event file found for {segment}")
    
    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)
        print(f"Total events loaded: {len(combined_events)}")
        return combined_events
    else:
        print(f"No events found for {toxin} {data_type} {segment}")
        return None

def calculate_event_rates_for_toxin(toxin, min_duration=5, max_duration=50, include_post_folders=False):
    """
    Calculate event rates for a specific toxin
    Counts ALL cells including those with zero events
    """
    results = []
    
    for data_type in ['voltage', 'calcium']:
        for segment in ['pre', 'post']:
            # Load event data
            events = load_event_data_by_toxin(toxin, data_type, segment, include_post_folders=include_post_folders)
            
            # Filter events by duration
            if events is not None:
                events = events[
                    (events['duration_sec'] > min_duration) & 
                    (events['duration_sec'] < max_duration)
                ]
                print(f"After duration filter ({min_duration}s-{max_duration}s): {len(events)} events")
            
            # Load timeseries data to get ALL cells
            timeseries_data, trials = load_timeseries_data_by_toxin(toxin, data_type, segment, include_post_folders=include_post_folders)
            
            if timeseries_data is None:
                print(f"No timeseries data for {data_type} {segment} - skipping")
                continue
            
            # Get total frames (5 Hz sampling rate)
            timeseries_cols = [col for col in timeseries_data.columns 
                             if col not in ['trial_string', 'cell_index', 'folder_source', 'global_cell_index']]
            total_frames = len(timeseries_cols)
            total_time_sec = total_frames / 5.0
            
            # Iterate through ALL cells in timeseries (including cells with zero events)
            for idx, row in timeseries_data.iterrows():
                trial = row['trial_string']
                folder = row['folder_source']
                cell_idx = row['cell_index']
                
                event_count = 0
                
                if events is not None and len(events) > 0:
                    # Count positive events for this cell in this trial AND folder
                    cell_events = events[
                        (events['trial_string'] == trial) & 
                        (events['folder_source'] == folder) &
                        (events['cell_index'] == cell_idx) & 
                        (events['event_type'] == 'positive')
                    ]
                    event_count = len(cell_events)
                
                # Calculate event rate per 100 seconds
                event_rate_per_100s = (event_count / total_time_sec) * 100
                
                results.append({
                    'toxin': toxin,
                    'data_type': data_type,
                    'segment': segment,
                    'event_rate_per_100s': event_rate_per_100s,
                    'trial_string': trial,
                    'folder_source': folder,
                    'cell_index': cell_idx,
                    'total_events': event_count,
                    'total_time_sec': total_time_sec
                })
            
            print(f"Processed {len(timeseries_data)} cells (including zero-event cells)")
    
    return pd.DataFrame(results)

def calculate_event_rates_for_multiple_toxins(toxins, min_duration=5, max_duration=50, include_post_folders=False):
    """
    Calculate event rates for multiple toxins
    """
    all_toxin_data = []
    
    for toxin in toxins:
        print(f"\n{'='*70}")
        print(f"Processing toxin: {toxin}")
        print(f"{'='*70}")
        
        toxin_data = calculate_event_rates_for_toxin(toxin, min_duration, max_duration, include_post_folders=include_post_folders)
        
        if toxin_data is not None and len(toxin_data) > 0:
            all_toxin_data.append(toxin_data)
            print(f"✓ {toxin}: {len(toxin_data)} cells processed")
        else:
            print(f"✗ {toxin}: No data found")
    
    if all_toxin_data:
        combined_data = pd.concat(all_toxin_data, ignore_index=True)
        return combined_data
    else:
        return None

def plot_multiple_toxins_comparison(event_data, toxins, save_path=None, stat_test='bootstrap', draw_toxin_bars=True):
    """
    Create plots comparing multiple toxins side-by-side with improved labeling
    """
    # Use default seaborn color palette (tab10 colors)
    default_colors = sns.color_palette("tab10", n_colors=len(toxins))
    toxin_colors = {toxin: default_colors[i] for i, toxin in enumerate(toxins)}
    
    plt.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 2, figsize=(8 + 4*len(toxins), 6))
    
    for i, data_type in enumerate(['voltage', 'calcium']):
        ax = axes[i]
        
        # Filter data for this data type
        type_data = event_data[event_data['data_type'] == data_type].copy()
        
        if len(type_data) == 0:
            ax.text(0.5, 0.5, f'No data for {data_type}', ha='center', va='center', transform=ax.transAxes)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            continue
        
        # Create a combined column for x-axis grouping
        type_data['toxin_segment'] = type_data['toxin'] + '_' + type_data['segment']
        
        # Define the order for x-axis: pre1, post1, pre2, post2, ...
        x_order = []
        for toxin in toxins:
            x_order.append(f"{toxin}_pre")
            x_order.append(f"{toxin}_post")
        
        # Filter to only include combinations that exist in the data
        x_order = [x for x in x_order if x in type_data['toxin_segment'].unique()]
        
        # Assign colors based on toxin
        type_data['color'] = type_data['toxin'].map(toxin_colors)
        
        # Create palette for seaborn (maps toxin_segment to color)
        palette = {}
        for x in x_order:
            toxin = '_'.join(x.split('_')[:-1])
            palette[x] = toxin_colors[toxin]
        
        # Create half violin plot with toxin-specific colors
        sns.violinplot(
            data=type_data,
            x='toxin_segment',
            y='event_rate_per_100s',
            ax=ax,
            order=x_order,
            palette=palette,
            inner=None,
            cut=0,
            linewidth=2,
            saturation=0.8,
            alpha=0.5,
            width=0.8,
            scale='width'
        )
        
        # Manually adjust violin positions to be half-width
        for collection in ax.collections:
            if hasattr(collection, 'get_paths'):
                paths = collection.get_paths()
                for path in paths:
                    vertices = path.vertices
                    x_center = vertices[:, 0].mean()
                    mask = vertices[:, 0] >= x_center
                    vertices[~mask, 0] = x_center
        
        # Overlay swarm plot with same colors - USING SEABORN for proper spacing
        sns.swarmplot(
            data=type_data,
            x='toxin_segment',
            y='event_rate_per_100s',
            ax=ax,
            size=10,
            order=x_order,
            palette=palette,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Set title based on data type
        if data_type == 'voltage':
            ax.set_title('Voltage Hyperpolarization Event Rate')
        else:
            ax.set_title('Calcium Event Rate')
        
        ax.set_ylabel('Events per 100s')
        ax.set_xlabel('')
        
        # Set x-axis properties
        ax.set_xticks(range(len(x_order)))
        
        # Simple Pre/With labels (no rotation)
        x_labels = []
        for x in x_order:
            segment = x.split('_')[-1]
            if segment == 'pre':
                x_labels.append('Pre')
            else:
                x_labels.append('With')
        
        ax.set_xticklabels(x_labels, rotation=0)
        
        # Group positions by toxin
        toxin_positions = {}
        for idx, x in enumerate(x_order):
            toxin = '_'.join(x.split('_')[:-1])
            if toxin not in toxin_positions:
                toxin_positions[toxin] = []
            toxin_positions[toxin].append(idx)
        
        # Add toxin name labels below x-axis
        for toxin in toxins:
            if toxin in toxin_positions:
                positions = toxin_positions[toxin]
                center_pos = np.mean(positions)
                toxin_label = TOXIN_LABELS.get(toxin, toxin)
                
                # Add text below
                ax.text(center_pos, -0.15, toxin_label, 
                       transform=ax.get_xaxis_transform(),
                       ha='center', va='top', fontsize=18,
                       color=toxin_colors[toxin], weight='bold')
                
                # Optional: Draw horizontal bar
                if draw_toxin_bars:
                    min_pos = min(positions) - 0.4
                    max_pos = max(positions) + 0.4
                    ax.plot([min_pos, max_pos], [-0.11, -0.11], 
                           transform=ax.get_xaxis_transform(),
                           color=toxin_colors[toxin], linewidth=3, 
                           solid_capstyle='round')
        
        # Statistical comparisons for each toxin (pre vs post)
        pairs = []
        for toxin in toxins:
            pre_key = f"{toxin}_pre"
            post_key = f"{toxin}_post"
            if pre_key in x_order and post_key in x_order:
                pairs.append((pre_key, post_key))
        
        if len(pairs) > 0:
            if stat_test == 'bootstrap':
                pvalues = []
                for pre_key, post_key in pairs:
                    pre_data = type_data[type_data['toxin_segment'] == pre_key]['event_rate_per_100s'].values
                    post_data = type_data[type_data['toxin_segment'] == post_key]['event_rate_per_100s'].values
                    
                    if len(pre_data) > 0 and len(post_data) > 0:
                        p_value = statsf.bootstrap_test_2sided(pre_data, post_data)[0]
                        pvalues.append(p_value)
                        print(f"{data_type} - {pre_key.split('_')[0]}: p={p_value:.3e}")
                    else:
                        pvalues.append(1.0)
                
                annotator = Annotator(ax, pairs, data=type_data, x='toxin_segment', y='event_rate_per_100s', order=x_order)
                annotator.configure(text_format='simple')
                annotator.set_pvalues(pvalues).annotate()
                
            elif stat_test == 'mann-whitney':
                annotator = Annotator(ax, pairs, data=type_data, x='toxin_segment', y='event_rate_per_100s', order=x_order)
                annotator.configure(test='Mann-Whitney', text_format='simple', show_test_name=False)
                annotator.apply_and_annotate()
    
    plt.tight_layout()
    # Add extra space at bottom for toxin labels
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        eps_path = str(save_path).replace('.png', '.eps')
        plt.savefig(eps_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"\nMulti-toxin Figure saved to: {save_path}")
        print(f"Multi-toxin EPS saved to: {eps_path}")
    
    plt.show()

def save_multiple_toxins_info(event_data, toxins, filename='multiple_toxins_info.txt'):
    """
    Save summary information for multiple toxins
    """
    info_path = save_dir / filename
    
    with open(info_path, 'w') as f:
        f.write(f"Multi-Toxin Event Rate Analysis\n")
        f.write("="*60 + "\n\n")
        
        for toxin in toxins:
            f.write(f"\n{'='*60}\n")
            f.write(f"TOXIN: {toxin}\n")
            f.write(f"{'='*60}\n\n")
            
            toxin_data = event_data[event_data['toxin'] == toxin]
            
            if len(toxin_data) == 0:
                f.write("No data available for this toxin\n\n")
                continue
            
            for data_type in ['voltage', 'calcium']:
                f.write(f"{data_type.upper()}\n")
                f.write("-"*60 + "\n")
                
                type_data = toxin_data[toxin_data['data_type'] == data_type]
                
                for segment in ['pre', 'post']:
                    segment_data = type_data[type_data['segment'] == segment]
                    
                    if len(segment_data) > 0:
                        n_cells = len(segment_data)
                        n_trials = segment_data['trial_string'].nunique()
                        n_folders = segment_data['folder_source'].nunique() if 'folder_source' in segment_data.columns else n_trials
                        mean_rate = segment_data['event_rate_per_100s'].mean()
                        std_rate = segment_data['event_rate_per_100s'].std()
                        n_zero_events = len(segment_data[segment_data['total_events'] == 0])
                        
                        # Extract slip and area counts
                        if 'folder_source' in segment_data.columns:
                            folder_sources = segment_data['folder_source'].unique()
                            
                            # Get unique slips (first 2 subparts)
                            slips = set()
                            for folder in folder_sources:
                                parts = folder.split('_')
                                if len(parts) >= 2:
                                    slip_id = '_'.join(parts[:2])
                                    slips.add(slip_id)
                            n_slips = len(slips)
                            
                            # Get unique areas (first 3 subparts)
                            areas = set()
                            for folder in folder_sources:
                                parts = folder.split('_')
                                if len(parts) >= 3:
                                    area_id = '_'.join(parts[:3])
                                    areas.add(area_id)
                            n_areas = len(areas)
                        else:
                            n_slips = 0
                            n_areas = 0
                        
                        f.write(f"{segment.upper()}:\n")
                        f.write(f"  Number of trials: {n_trials}\n")
                        f.write(f"  Number of folders: {n_folders}\n")
                        f.write(f"  Number of slips: {n_slips}\n")
                        f.write(f"  Number of areas: {n_areas}\n")
                        f.write(f"  Number of cells: {n_cells}\n")
                        f.write(f"  Cells with zero events: {n_zero_events}\n")
                        f.write(f"  Mean event rate: {mean_rate:.3f} ± {std_rate:.3f} events/100s\n")
                        
                        # List folders
                        if 'folder_source' in segment_data.columns:
                            folder_list = sorted(segment_data['folder_source'].unique())
                            f.write(f"  Folders: {', '.join(folder_list)}\n\n")
                        else:
                            trial_list = sorted(segment_data['trial_string'].unique())
                            f.write(f"  Trials: {', '.join(trial_list)}\n\n")
                    else:
                        f.write(f"{segment.upper()}:\n")
                        f.write(f"  No data available\n\n")
                
                # Calculate p-value
                pre_data = type_data[type_data['segment'] == 'pre']['event_rate_per_100s'].values
                post_data = type_data[type_data['segment'] == 'post']['event_rate_per_100s'].values
                
                if len(pre_data) > 0 and len(post_data) > 0:
                    p_value = statsf.bootstrap_test_2sided(pre_data, post_data)[0]
                    f.write(f"  P-value (bootstrap, 2-sided): {p_value:.6f}\n")
                    
                    if len(pre_data) != len(post_data):
                        f.write(f"  Note: Unpaired comparison (n_pre={len(pre_data)}, n_post={len(post_data)})\n")
                else:
                    f.write(f"  P-value: Not enough data for comparison\n")
                
                f.write("\n")
    
    print(f"Multi-toxin info saved to: {info_path}")

def analyze_multiple_toxins(toxins, min_duration=5, max_duration=50, draw_toxin_bars=True, include_post_folders=False):
    """
    Complete analysis for multiple toxins
    """
    print(f"\n{'='*70}")
    print(f"Analyzing multiple toxins: {', '.join(toxins)}")
    if include_post_folders:
        print(f"  (include_post_folders=True)")
    print(f"{'='*70}")
    
    # Calculate event rates for all toxins
    event_data = calculate_event_rates_for_multiple_toxins(toxins, min_duration, max_duration, include_post_folders=include_post_folders)
    
    if event_data is None or len(event_data) == 0:
        print(f"\n✗ No event data found for any toxin")
        return None
    
    print(f"\n{'='*70}")
    print(f"Overall Summary:")
    print(f"  Total cells analyzed: {len(event_data)}")
    print(f"  Cells with zero events: {len(event_data[event_data['total_events'] == 0])}")
    print(f"  Total toxins: {event_data['toxin'].nunique()}")
    print(f"  Total trials: {event_data['trial_string'].nunique()}")
    if 'folder_source' in event_data.columns:
        print(f"  Total folders: {event_data['folder_source'].nunique()}")
    
    for toxin in toxins:
        toxin_data = event_data[event_data['toxin'] == toxin]
        if len(toxin_data) > 0:
            print(f"  {toxin}: {len(toxin_data)} cells")
    
    print(f"{'='*70}")
    
    # Create filename for multi-toxin plot
    toxins_str = '_'.join(toxins)
    save_path = save_dir / f'multiple_toxins_{toxins_str}_event_rate_comparison.png'
    
    # Plot comparison
    plot_multiple_toxins_comparison(event_data, toxins, save_path, draw_toxin_bars=draw_toxin_bars)
    
    # Save info
    save_multiple_toxins_info(event_data, toxins, filename=f'multiple_toxins_{toxins_str}_info.txt')
    
    return event_data

# Example usage
if __name__ == "__main__":
    # Define multiple toxins to compare
    #toxins = ['Ca_free', 'Thapsigargin', 'dantrolene']
    #toxins = ['L-15','DMSO_0.1%']
    #toxins = ['A01','Ani9']
    toxins = ['Nifepidine', 'TTA-A2','YM58483','cbx']
    #toxins = ['CBA','BAY1797','Carbachol']
    
    # Analyze all toxins together
    # Set draw_toxin_bars=True to show horizontal bars, False to hide them
    # Set include_post_folders=True to include _post folder data IN ADDITION to normal folders
    try:
        event_data = analyze_multiple_toxins(
            toxins, 
            min_duration=5, 
            max_duration=50, 
            draw_toxin_bars=True,
            include_post_folders=True
        )
    except Exception as e:
        print(f"\n✗ Error analyzing toxins: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")