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
import ptitprince as pt

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
#df_path = dataframes_dir / 'MDA_MB_468_segmented_results4.csv'
df_path = dataframes_dir / 'long_acqs_MDA_MB_468_all.csv'
save_dir = data_dir / 'toxin_analysis_plots'
save_dir.mkdir(parents=True, exist_ok=True)

def load_timeseries_data_by_toxin(toxin, data_type='voltage', segment='pre', include_post_folders=False):
    """
    Load and concatenate timeseries data for a specific toxin, data type, and segment
    
    Parameters:
    -----------
    toxin : str
        Toxin name
    data_type : str
        'voltage' or 'calcium'
    segment : str
        'pre' or 'post'
    include_post_folders : bool
        If True and segment='post', also loads from _post folders in addition to normal folders
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n_focus']
    
    # Filter by toxin
    toxin_df = df[df['expt'].str.contains(toxin, case=False, na=False)]
    #toxin_df = df[df['expt'] == toxin]
    
    print(f"\n=== Loading TIMESERIES {data_type} {segment} for toxin: {toxin} ===")
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
    Load and process event data for a specific toxin, data type, and segment
    
    Parameters:
    -----------
    toxin : str
        Toxin name to search for in metadata
    data_type : str
        'voltage' or 'calcium'
    segment : str
        'pre' or 'post'
    include_post_folders : bool
        If True and segment='post', also loads from _post folders in addition to normal folders
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    #df = df[df['use'] != 'n_focus']
    
    # Filter by toxin
    toxin_df = df[df['expt'].str.contains(toxin, case=False, na=False)]
    #toxin_df = df[df['expt'] == toxin]
    
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
                        # Remove duplicates
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
        return combined_events
    else:
        return None

def calculate_std_for_toxin(toxin, include_post_folders=False):
    """
    Calculate standard deviation for voltage and calcium data (pre and post) for a specific toxin
    
    Parameters:
    -----------
    toxin : str
        Toxin name
    include_post_folders : bool
        If True, also loads from _post folders for post segments
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with std values for plotting
    """
    results = []
    
    for data_type in ['voltage', 'calcium']:
        for segment in ['pre', 'post']:
            data, trials = load_timeseries_data_by_toxin(toxin, data_type, segment, include_post_folders=include_post_folders)
            
            if data is not None:
                # Calculate std for each cell (row)
                timeseries_cols = [col for col in data.columns if col not in ['trial_string', 'cell_index', 'folder_source', 'global_cell_index']]
                std_values = data[timeseries_cols].std(axis=1)
                
                # Create result dataframe
                for i, std_val in enumerate(std_values):
                    results.append({
                        'toxin': toxin,
                        'data_type': data_type,
                        'segment': segment,
                        'std_value': std_val,
                        'trial_string': data.iloc[i]['trial_string'],
                        'folder_source': data.iloc[i]['folder_source'],
                        'cell_index': data.iloc[i]['cell_index']
                    })
            else:
                print(f"No data found for {toxin} {data_type} {segment}")
    
    return pd.DataFrame(results)

def calculate_event_rates_for_toxin(toxin, min_duration=5, max_duration=50, include_post_folders=False):
    """
    Calculate event rates for voltage and calcium data (pre and post) for a specific toxin
    
    Parameters:
    -----------
    toxin : str
        Toxin name
    min_duration : float
        Minimum event duration in seconds (default: 5)
    max_duration : float
        Maximum event duration in seconds (default: 50)
    include_post_folders : bool
        If True, also loads from _post folders for post segments
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with event rates for plotting
    """
    results = []
    
    for data_type in ['voltage', 'calcium']:
        for segment in ['pre', 'post']:
            # Load event data
            events = load_event_data_by_toxin(toxin, data_type, segment, include_post_folders=include_post_folders)
            
            # Filter events: only keep events with duration > min_duration and < max_duration
            if events is not None:
                events = events[
                    (events['duration_sec'] > min_duration) & 
                    (events['duration_sec'] < max_duration)
                ]
                print(f"After duration filter ({min_duration}s-{max_duration}s): {len(events)} events")
            
            # Load timeseries data to get total cell count and frame count
            timeseries_data, trials = load_timeseries_data_by_toxin(toxin, data_type, segment, include_post_folders=include_post_folders)
            
            if timeseries_data is not None:
                # Get total frames (5 Hz sampling rate)
                timeseries_cols = [col for col in timeseries_data.columns 
                                 if col not in ['trial_string', 'cell_index', 'folder_source', 'global_cell_index']]
                total_frames = len(timeseries_cols)
                total_time_sec = total_frames / 5.0  # 5 Hz sampling
                
                # Iterate through ALL cells in timeseries (not just cells with events)
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

def plot_std_comparison(std_data, toxin, save_path=None, stat_test='bootstrap'):
    """
    Create swarm plots comparing pre vs post standard deviation
    
    Parameters:
    -----------
    std_data : pandas.DataFrame
        DataFrame with std values
    toxin : str
        Toxin name for title
    save_path : str or Path, optional
        Path to save the figure
    stat_test : str
        'mann-whitney' or 'bootstrap' for statistical testing
    """
    plt.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, data_type in enumerate(['voltage', 'calcium']):
        ax = axes[i]
        
        # Filter data for this data type
        type_data = std_data[std_data['data_type'] == data_type].copy()
        
        if len(type_data) > 0:
            # Create swarm plot
            sns.stripplot(data=type_data, x='segment', y='std_value', ax=ax, size=10, order=['pre', 'post'])
            
            # Remove top and right spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Set labels with proper mathematical notation
            ax.set_xlabel("")
            if data_type == 'voltage':
                ax.set_ylabel(r'$\sigma_{V}$')
            else:  # calcium
                ax.set_ylabel(r'$\sigma_{Ca^{2+}}$')
            
            formatted_toxin = toxin.replace('_', ' ').replace('uM', r'$\mu$M').replace('mM', ' mM')
            # Set x-tick labels with formatted toxin name
            parts = formatted_toxin.split()
            if len(parts) >= 2:
                ax.set_xticklabels(['Pre', f'With {parts[-1]} {parts[-2]}'])
            else:
                ax.set_xticklabels(['Pre', f'With {formatted_toxin}'])
            
            # Statistical comparison
            pre_data = type_data[type_data['segment'] == 'pre']['std_value'].values
            post_data = type_data[type_data['segment'] == 'post']['std_value'].values
            
            if len(pre_data) > 0 and len(post_data) > 0:
                pairs = [('pre', 'post')]
                
                if stat_test == 'bootstrap':
                    # Calculate bootstrap p-value
                    p_value = statsf.bootstrap_test(pre_data, post_data)[0]
                    pvalues = [p_value]
                    
                    # Use Annotator to display bootstrap results
                    annotator = Annotator(ax, pairs, data=type_data, x='segment', y='std_value')
                    annotator.configure(text_format='simple')
                    annotator.set_pvalues(pvalues).annotate()
                    
                elif stat_test == 'mann-whitney':
                    # Use Mann-Whitney test
                    annotator = Annotator(ax, pairs, data=type_data, x='segment', y='std_value')
                    annotator.configure(test='Mann-Whitney', text_format='simple', show_test_name=False)
                    annotator.apply_and_annotate()
        else:
            ax.text(0.5, 0.5, f'No data for {data_type}', ha='center', va='center', transform=ax.transAxes)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        # Also save as EPS
        eps_path = str(save_path).replace('.png', '.eps')
        plt.savefig(eps_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
        print(f"EPS saved to: {eps_path}")
    
    plt.show()

def plot_event_rate_comparison(event_data, toxin, save_path=None, stat_test='bootstrap'):
    """
    Create half violin + swarm plots comparing pre vs post event rates
    
    Parameters:
    -----------
    event_data : pandas.DataFrame
        DataFrame with event rates
    toxin : str
        Toxin name for title
    save_path : str or Path, optional
        Path to save the figure
    stat_test : str
        'mann-whitney' or 'bootstrap' for statistical testing
    """
    plt.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simple toxin name formatting
    formatted_toxin = toxin.replace('_', ' ').replace('uM', r'$\mu$M').replace('mM', ' mM')
    
    for i, data_type in enumerate(['voltage', 'calcium']):
        ax = axes[i]
        
        # Filter data for this data type
        type_data = event_data[event_data['data_type'] == data_type].copy()
        
        if len(type_data) > 0:
            # Create half violin plot
            sns.violinplot(
                data=type_data, 
                x='segment', 
                y='event_rate_per_100s', 
                ax=ax,
                order=['pre', 'post'],
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
            
            # Overlay swarm plot
            sns.swarmplot(
                data=type_data, 
                x='segment', 
                y='event_rate_per_100s', 
                ax=ax,
                size=10,
                order=['pre', 'post'],
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
            
            # Set x-tick labels with formatted toxin name
            parts = formatted_toxin.split()
            if len(parts) >= 2:
                ax.set_xticklabels(['Pre', f'With {parts[-1]} {parts[-2]}'])
            else:
                ax.set_xticklabels(['Pre', f'With {formatted_toxin}'])
            
            # Manual formatting for specific toxins
            toxin_labels = {
                '4AP': ['Pre', 'With 5 mM 4-AP'],
                'Ca_free': ['Pre', r'With Ca$^{2+}$ Free External'],
                'ATP': ['Pre', 'With 1 mM ATP-$\gamma$S'],
                'cbx_100uM': ['Pre', 'With 100 $\mu$M CBX'],
                'heparin': ['Pre', 'With 5 mg/mL Heparin'],
                'TRAM-34_1uM_JEDIonly': ['Pre', 'With 1 $\mu$M TRAM-34'],
                'TTA-A2': ['Pre', 'With 10 $\mu$M TTA-A2'],
                'Carbachol': ['Pre', 'With 100 $\mu$M Carbachol'],
                'CBA': ['Pre', 'With 50 $\mu$M CBA'],
                'BAY1797': ['Pre', 'With 10 $\mu$M BAY1797'],
                'S-Bayk': ['Pre', 'With 10 $\mu$M S-Bayk'],
                'YM58483' : ['Pre', 'With 10 $\mu$M YM58483']
            }
            
            if toxin in toxin_labels:
                ax.set_xticklabels(toxin_labels[toxin])
            
            # Statistical comparison
            pre_data = type_data[type_data['segment'] == 'pre']['event_rate_per_100s'].values
            post_data = type_data[type_data['segment'] == 'post']['event_rate_per_100s'].values
            
            if len(pre_data) > 0 and len(post_data) > 0:
                pairs = [('pre', 'post')]
                
                if stat_test == 'bootstrap':
                    p_value = statsf.bootstrap_test_2sided(pre_data, post_data)[0]
                    pvalues = [p_value]
                    annotator = Annotator(ax, pairs, data=type_data, x='segment', y='event_rate_per_100s')
                    annotator.configure(text_format='simple')
                    annotator.set_pvalues(pvalues).annotate()
                    
                elif stat_test == 'mann-whitney':
                    annotator = Annotator(ax, pairs, data=type_data, x='segment', y='event_rate_per_100s')
                    annotator.configure(test='Mann-Whitney', text_format='simple', show_test_name=False)
                    annotator.apply_and_annotate()
        else:
            ax.text(0.5, 0.5, f'No data for {data_type}', ha='center', va='center', transform=ax.transAxes)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        eps_path = str(save_path).replace('.png', '.eps')
        plt.savefig(eps_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Event Rate Figure saved to: {save_path}")
        print(f"Event Rate EPS saved to: {eps_path}")
    
    plt.show()

def save_event_rate_info(event_data, toxin, save_dir):
    """
    Save trial counts, cell counts, mean event rates, and p-values to a text file
    
    Parameters:
    -----------
    event_data : pandas.DataFrame
        DataFrame with event rates
    toxin : str
        Toxin name
    save_dir : Path
        Directory to save the info file
    """
    info_path = save_dir / f'events_info_{toxin}.txt'
    
    with open(info_path, 'w') as f:
        f.write(f"Event Rate Analysis for {toxin}\n")
        f.write("="*60 + "\n\n")
        
        for data_type in ['voltage', 'calcium']:
            f.write(f"{data_type.upper()}\n")
            f.write("-"*60 + "\n")
            
            type_data = event_data[event_data['data_type'] == data_type]
            
            for segment in ['pre', 'post']:
                segment_data = type_data[type_data['segment'] == segment]
                
                if len(segment_data) > 0:
                    n_cells = len(segment_data)
                    n_folders = segment_data['folder_source'].nunique()
                    n_trials = segment_data['trial_string'].nunique()
                    mean_rate = segment_data['event_rate_per_100s'].mean()
                    std_rate = segment_data['event_rate_per_100s'].std()
                    n_zero_events = len(segment_data[segment_data['total_events'] == 0])
                    
                    # Extract slip and area counts
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
                    
                    f.write(f"{segment.upper()}:\n")
                    f.write(f"  Number of trials: {n_trials}\n")
                    f.write(f"  Number of folders: {n_folders}\n")
                    f.write(f"  Number of slips: {n_slips}\n")
                    f.write(f"  Number of areas: {n_areas}\n")
                    f.write(f"  Number of cells: {n_cells}\n")
                    f.write(f"  Cells with zero events: {n_zero_events}\n")
                    f.write(f"  Mean event rate: {mean_rate:.3f} ± {std_rate:.3f} events/100s\n")
                    
                    # List folders
                    folder_list = sorted(segment_data['folder_source'].unique())
                    f.write(f"  Folders: {', '.join(folder_list)}\n\n")
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
    
    print(f"Event info saved to: {info_path}")

def analyze_toxin(toxin, save_dir=None, plot_std=True, plot_event_rate=True, include_post_folders=False):
    """
    Complete analysis for a specific toxin
    
    Parameters:
    -----------
    toxin : str
        Toxin name to analyze
    save_dir : str or Path, optional
        Directory to save figures
    plot_std : bool
        Whether to plot standard deviation comparison
    plot_event_rate : bool
        Whether to plot event rate comparison
    include_post_folders : bool
        If True, also loads from _post folders for post segments (in addition to normal folders)
    """
    print(f"\n{'='*70}")
    print(f"Analyzing toxin: {toxin}")
    if include_post_folders:
        print(f"  (include_post_folders=True)")
    print(f"{'='*70}")

    std_data = None
    event_data = None
    
    if plot_std:
        print("\nLoading standard deviation data...")
        std_data = calculate_std_for_toxin(toxin, include_post_folders=include_post_folders)

    if plot_event_rate:
        print("\nLoading event rate data...")
        event_data = calculate_event_rates_for_toxin(toxin, min_duration=5, max_duration=50, include_post_folders=include_post_folders)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        std_save_path = save_dir / f'{toxin}_std_comparison.png'
        event_save_path = save_dir / f'{toxin}_event_rate_comparison.png'
    else:
        std_save_path = None
        event_save_path = None
    
    if plot_std and std_data is not None and len(std_data) > 0:
        plot_std_comparison(std_data, toxin, std_save_path)
    elif plot_std:
        print(f"No standard deviation data found for {toxin}")
    
    if plot_event_rate and event_data is not None and len(event_data) > 0:
        plot_event_rate_comparison(event_data, toxin, event_save_path)
        
        # Save event rate info to text file
        if save_dir:
            save_event_rate_info(event_data, toxin, save_dir)
    elif plot_event_rate:
        print(f"No event rate data found for {toxin}")
    
    return std_data, event_data

# Example usage
if __name__ == "__main__":
    # Define toxins to analyze
    toxins = ['4AP','A01','Ani9','Ca_free','cbx','dantrolene','DMSO','L-15','Thapsigargin']
    toxins = ['ATP','BAY1797','Carbachol','heparin','Nifepidine','PPADS','S-Bayk']
    toxins = ['TRAM-34_1uM','TRAM-34_1uM_JEDIonly'] # need to revise the df filter condition for these
    toxins = ['Thapsigargin','Nifepidine','YM58483','BAY1797']
    # Create save directory
    save_dir = data_dir / 'toxin_analysis_plots'
    
    # Analyze each toxin
    all_results = {}
    for toxin in toxins:
        try:
            # Set include_post_folders=True to include _post folder data IN ADDITION to normal folders
            # Set include_post_folders=False to only use normal folders
            std_data, event_data = analyze_toxin(
                toxin, 
                save_dir, 
                plot_std=False, 
                plot_event_rate=True,
                include_post_folders=True
            )
            all_results[toxin] = {'std_data': std_data, 'event_data': event_data}
        except Exception as e:
            print(f"\n✗ Error analyzing {toxin}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"Successfully analyzed {len(all_results)} toxins")
    print(f"{'='*70}")