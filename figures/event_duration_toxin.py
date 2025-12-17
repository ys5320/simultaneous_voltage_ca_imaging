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
df_path = dataframes_dir / 'MDA_MB_468_segmented_results4.csv'
save_dir = data_dir / 'toxin_analysis_plots'
save_dir.mkdir(parents=True, exist_ok=True)

# Toxin display labels
TOXIN_LABELS = {
    'Carbachol': r'100 $\mu$M Carbachol',
    'TTA-A2': r'10 $\mu$M TTA-A2',
    'CBA': r'50 $\mu$M CBA',
    'BAY1797': r'10 $\mu$M BAY1797',
    'S-Bayk': r'10 $\mu$M S-Bayk',
    'cbx_100uM': r'100 $\mu$M CBX',
    '4AP': '5 mM 4-AP',
    'Ca_free': r'Ca$^{2+}$ Free',
    'ATP_1mM': '1 mM ATP',
    'heparin': '5 mg/mL Heparin',
    'TRAM-34': r'1 $\mu$M TRAM-34',
    'Nifedipine': r'10 $\mu$M Nifedipine',
    'Ani9_10uM': r'10 $\mu$M Ani9',
    'dantrolene_10uM': r'10 $\mu$M Dantrolene',
    'Thapsigargin_1uM': r'1 $\mu$M Thapsigargin',
}

def load_event_data_by_toxin(toxin, data_type='voltage', segment='pre', include_post_folders=False):
    """
    Load event data from individual trial folders
    
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
    
    Returns:
    --------
    pandas.DataFrame or None
    """
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n_focus']
    
    # Get ALL trials (paired + unpaired)
    toxin_df = df[df['expt'].str.contains(toxin, case=False, na=False)]
    
    print(f"\n=== Loading EVENT data: {data_type} {segment} for toxin: {toxin} ===")
    print(f"Found {len(toxin_df)} trial entries in metadata")
    if include_post_folders and segment == 'post':
        print(f"  (include_post_folders=True: will load from BOTH normal and _post folders)")
    
    all_events = []
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
            
            if not trial_dir.exists():
                print(f"✗ {trial_string} (folder: {folder_name}): Folder not found")
                continue
            
            # Try different possible event file patterns
            # If using _post folder, the filename also has _post in the trial_string
            if folder_name.endswith('_post'):
                possible_patterns = [
                    f"events_{data_type}_{segment}_{trial_string}_post_simple_QC_final.csv",
                    f"events_{data_type}_{segment}_{trial_string}_post_QC_final.csv",
                ]
            else:
                possible_patterns = [
                    f"events_{data_type}_{segment}_{trial_string}_simple_QC_final.csv",
                    f"events_{data_type}_{segment}_{trial_string}_QC_final.csv",
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
                        successful_trials.append(folder_name)
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

def calculate_event_durations_for_toxin(toxin, min_duration=5, max_duration=50, include_post_folders=False):
    """
    Calculate event durations for a specific toxin
    
    Parameters:
    -----------
    toxin : str
        Toxin name
    min_duration : float
        Minimum event duration in seconds (default: 5)
    max_duration : float
        Maximum event duration in seconds (default: 50)
    include_post_folders : bool
        If True, also looks for POST events in folders with _post suffix
    
    Returns:
    --------
    pandas.DataFrame with event durations
    """
    results = []
    
    for data_type in ['voltage', 'calcium']:
        for segment in ['pre', 'post']:
            # Load events (pass include_post_folders parameter)
            events_df = load_event_data_by_toxin(toxin, data_type, segment, include_post_folders=include_post_folders)
            
            if events_df is None or len(events_df) == 0:
                print(f"No events data for {data_type} {segment} - skipping")
                continue
            
            # Filter events by duration
            events_filtered = events_df[
                (events_df['duration_sec'] > min_duration) & 
                (events_df['duration_sec'] < max_duration)
            ]
            print(f"After duration filter ({min_duration}s-{max_duration}s): {len(events_filtered)} events")
            
            if len(events_filtered) == 0:
                print(f"No events remaining after filtering for {data_type} {segment}")
                continue
            
            # Filter for positive events only
            positive_events = events_filtered[events_filtered['event_type'] == 'positive']
            
            print(f"Positive events: {len(positive_events)}")
            
            # For each event, store its duration
            for idx, event in positive_events.iterrows():
                results.append({
                    'toxin': toxin,
                    'data_type': data_type,
                    'segment': segment,
                    'duration_sec': event['duration_sec'],
                    'trial_string': event['trial_string'],
                    'folder_source': event.get('folder_source', event['trial_string']),  # Track folder source
                    'cell_index': event['cell_index']
                })
    
    return pd.DataFrame(results)

def plot_event_duration_comparison(event_data, toxin, save_path=None, stat_test='bootstrap'):
    """
    Create half violin + swarm plots comparing pre vs post event durations
    """
    plt.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, data_type in enumerate(['voltage', 'calcium']):
        ax = axes[i]
        
        # Filter data for this data type
        type_data = event_data[event_data['data_type'] == data_type].copy()
        
        if len(type_data) > 0:
            # Create half violin plot
            sns.violinplot(
                data=type_data, 
                x='segment', 
                y='duration_sec', 
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
                        vertices[~mask, 0] = x_center - 0.15
            
            # Overlay swarm plot
            sns.swarmplot(
                data=type_data, 
                x='segment', 
                y='duration_sec', 
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
                ax.set_title('Voltage Hyperpolarization Event Duration')
            else:
                ax.set_title('Calcium Event Duration')
            
            ax.set_ylabel('Duration (s)')
            ax.set_xlabel('')
            
            # Set x-tick labels
            toxin_label = TOXIN_LABELS.get(toxin, toxin)
            ax.set_xticklabels(['Pre', f'With {toxin_label}'])
            
            # Statistical comparison
            pre_data = type_data[type_data['segment'] == 'pre']['duration_sec'].values
            post_data = type_data[type_data['segment'] == 'post']['duration_sec'].values
            
            if len(pre_data) > 0 and len(post_data) > 0:
                pairs = [('pre', 'post')]
                
                if stat_test == 'bootstrap':
                    p_value = statsf.bootstrap_test_2sided(pre_data, post_data)[0]
                    pvalues = [p_value]
                    annotator = Annotator(ax, pairs, data=type_data, x='segment', y='duration_sec')
                    annotator.configure(text_format='simple')
                    annotator.set_pvalues(pvalues).annotate()
                    print(f"{data_type} - pre vs post: p={p_value:.3e}")
                    
                elif stat_test == 'mann-whitney':
                    annotator = Annotator(ax, pairs, data=type_data, x='segment', y='duration_sec')
                    annotator.configure(test='Mann-Whitney', text_format='simple', show_test_name=False)
                    annotator.apply_and_annotate()
            else:
                if len(pre_data) == 0:
                    ax.text(0.5, 0.95, 'No PRE data', ha='center', va='top', 
                           transform=ax.transAxes, fontsize=12, color='red')
                elif len(post_data) == 0:
                    ax.text(0.5, 0.95, 'No POST data', ha='center', va='top', 
                           transform=ax.transAxes, fontsize=12, color='red')
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
        print(f"\nEvent Duration Figure saved to: {save_path}")
        print(f"Event Duration EPS saved to: {eps_path}")
    
    plt.show()

def save_event_duration_info(event_data, toxin):
    """
    Save event duration statistics to a text file
    """
    info_path = save_dir / f'events_duration_info_{toxin}.txt'
    
    with open(info_path, 'w') as f:
        f.write(f"Event Duration Analysis for {toxin}\n")
        f.write("="*60 + "\n\n")
        
        for data_type in ['voltage', 'calcium']:
            f.write(f"{data_type.upper()}\n")
            f.write("-"*60 + "\n")
            
            type_data = event_data[event_data['data_type'] == data_type]
            
            for segment in ['pre', 'post']:
                segment_data = type_data[type_data['segment'] == segment]
                
                if len(segment_data) > 0:
                    n_events = len(segment_data)
                    n_trials = segment_data['trial_string'].nunique()
                    n_folders = segment_data['folder_source'].nunique() if 'folder_source' in segment_data.columns else n_trials
                    mean_duration = segment_data['duration_sec'].mean()
                    std_duration = segment_data['duration_sec'].std()
                    median_duration = segment_data['duration_sec'].median()
                    
                    f.write(f"{segment.upper()}:\n")
                    f.write(f"  Number of trials: {n_trials}\n")
                    f.write(f"  Number of folders: {n_folders}\n")
                    f.write(f"  Number of events: {n_events}\n")
                    f.write(f"  Mean duration: {mean_duration:.3f} ± {std_duration:.3f} s\n")
                    f.write(f"  Median duration: {median_duration:.3f} s\n")
                    
                    # List folders if available
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
            pre_data = type_data[type_data['segment'] == 'pre']['duration_sec'].values
            post_data = type_data[type_data['segment'] == 'post']['duration_sec'].values
            
            if len(pre_data) > 0 and len(post_data) > 0:
                p_value = statsf.bootstrap_test_2sided(pre_data, post_data)[0]
                f.write(f"  P-value (bootstrap, 2-sided): {p_value:.6f}\n")
            else:
                f.write(f"  P-value: Not enough data for comparison\n")
            
            f.write("\n")
    
    print(f"Event duration info saved to: {info_path}")

def analyze_event_duration(toxin, min_duration=5, max_duration=50, include_post_folders=False):
    """
    Complete analysis for event durations of a specific toxin
    
    Parameters:
    -----------
    toxin : str
        Toxin name
    min_duration : float
        Minimum event duration
    max_duration : float
        Maximum event duration  
    include_post_folders : bool
        If True, also looks for POST events in folders with _post suffix
    """
    print(f"\n{'='*70}")
    print(f"Analyzing event durations for toxin: {toxin}")
    if include_post_folders:
        print(f"  (include_post_folders=True)")
    print(f"{'='*70}")
    
    # Calculate event durations (pass include_post_folders parameter)
    event_data = calculate_event_durations_for_toxin(toxin, min_duration, max_duration, include_post_folders=include_post_folders)
    
    if event_data is None or len(event_data) == 0:
        print(f"\n✗ No event data found for {toxin}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Summary for {toxin}:")
    print(f"  Total events analyzed: {len(event_data)}")
    print(f"  Total trials: {event_data['trial_string'].nunique()}")
    if 'folder_source' in event_data.columns:
        print(f"  Total folders: {event_data['folder_source'].nunique()}")
    
    # Check for balance between PRE and POST
    pre_count = len(event_data[event_data['segment'] == 'pre'])
    post_count = len(event_data[event_data['segment'] == 'post'])
    print(f"  PRE events: {pre_count}")
    print(f"  POST events: {post_count}")
    
    print(f"{'='*70}")
    
    # Plot event duration comparison
    save_path = save_dir / f'{toxin}_event_duration_comparison.png'
    plot_event_duration_comparison(event_data, toxin, save_path)
    
    # Save event duration info
    save_event_duration_info(event_data, toxin)
    
    return event_data

# Example usage
if __name__ == "__main__":
    # Analyze Carbachol
    toxin = 'Carbachol'
    
    try:
        # Set include_post_folders=True to include _post folder events IN ADDITION to normal folders
        # Set include_post_folders=False to only use normal folders
        event_data = analyze_event_duration(toxin, min_duration=5, max_duration=200, include_post_folders=True)
        if event_data is not None:
            print(f"\n✓ Successfully analyzed {toxin}")
    except Exception as e:
        print(f"\n✗ Error analyzing {toxin}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")