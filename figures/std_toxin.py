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
df_path = dataframes_dir / 'MDA_MB_468_dataframe_tc_extracted.csv'

def load_timeseries_data_by_toxin(toxin, data_type='voltage', segment='pre'):
    """
    Load and concatenate timeseries data for a specific toxin, data type, and segment
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n']
    
    # Filter by toxin
    toxin_trials = df[df['expt'].str.contains(toxin, case=False, na=False)]['trial_string'].unique()
    
    # Debug: Print filtered trials
    print(f"\n=== Loading {data_type} {segment} for toxin: {toxin} ===")
    print(f"Trials after filtering: {list(toxin_trials)}")
    
    all_data = []
    successful_trials = []
    
    for trial_string in toxin_trials:
        trial_dir = results_pipeline_dir / trial_string
        
        # Use glob to find matching files
        pattern = f"{segment}_{data_type}*{trial_string}_raw.csv"
        matching_files = list(trial_dir.glob(pattern))
        
        if matching_files:
            file_path = matching_files[0]  # Take the first match
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
                timeseries_data['trial_string'] = trial_string
                timeseries_data['cell_index'] = range(len(timeseries_data))
                
                all_data.append(timeseries_data)
                successful_trials.append(trial_string)
                print(f"✓ {trial_string}: Successfully loaded")
                
            except Exception as e:
                print(f"✗ {trial_string}: Error loading - {e}")
        else:
            print(f"✗ {trial_string}: File not found - {trial_dir / pattern}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data, successful_trials
    else:
        return None, []

def calculate_std_for_toxin(toxin):
    """
    Calculate standard deviation for voltage and calcium data (pre and post) for a specific toxin
    
    Parameters:
    -----------
    toxin : str
        Toxin name
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with std values for plotting
    """
    results = []
    
    for data_type in ['voltage', 'calcium']:
        for segment in ['pre', 'post']:
            data, trials = load_timeseries_data_by_toxin(toxin, data_type, segment)
            
            if data is not None:
                # Calculate std for each cell (row)
                timeseries_cols = [col for col in data.columns if col not in ['trial_string', 'cell_index']]
                std_values = data[timeseries_cols].std(axis=1)
                
                # Create result dataframe
                for i, std_val in enumerate(std_values):
                    results.append({
                        'toxin': toxin,
                        'data_type': data_type,
                        'segment': segment,
                        'std_value': std_val,
                        'trial_string': data.iloc[i]['trial_string'],
                        'cell_index': data.iloc[i]['cell_index']
                    })
            else:
                print(f"No data found for {toxin} {data_type} {segment}")
    
    return pd.DataFrame(results)

def calculate_event_rates_for_toxin(toxin):
    """
    Calculate event rates for voltage and calcium data (pre and post) for a specific toxin
    
    Parameters:
    -----------
    toxin : str
        Toxin name
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with event rates for plotting
    """
    results = []
    
    for data_type in ['voltage', 'calcium']:
        for segment in ['pre', 'post']:
            # Load event data
            events = load_event_data_by_toxin(toxin, data_type, segment)
            
            # Filter events: only keep events with duration > 5s
            if events is not None:
                events = events[events['duration_sec'] > 5]
                events = events[events['duration_sec'] < 50]
            
            # Load timeseries data to get total cell count and frame count
            timeseries_data, trials = load_timeseries_data_by_toxin(toxin, data_type, segment)
            
            if timeseries_data is not None:
                # Get total frames (5 Hz sampling rate)
                timeseries_cols = [col for col in timeseries_data.columns if col not in ['trial_string', 'cell_index']]
                total_frames = len(timeseries_cols)
                total_time_sec = total_frames / 5.0  # 5 Hz sampling
                
                # Iterate through ALL cells in timeseries (not just cells with events)
                for idx, row in timeseries_data.iterrows():
                    trial = row['trial_string']
                    cell_idx = row['cell_index']
                    
                    event_count = 0
                    
                    if events is not None and len(events) > 0:
                        # Count positive events for this cell in this trial with duration >= 5s
                        cell_events = events[
                            (events['trial_string'] == trial) & 
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
                        'cell_index': cell_idx,
                        'total_events': event_count,
                        'total_time_sec': total_time_sec
                    })
    
    return pd.DataFrame(results)

def load_event_data_by_toxin(toxin, data_type='voltage', segment='pre'):
    """
    Load and process event data for a specific toxin, data type, and segment
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n']
    
    # Filter by toxin
    toxin_trials = df[df['expt'].str.contains(toxin, case=False, na=False)]['trial_string'].unique()
    #toxin_trials = df[df['expt'] == toxin]['trial_string']
    #toxin_trials = df[df['expt'].str.lower() == toxin.lower()]['trial_string']
    
    # Debug: Print filtered trials
    print(f"\n=== Loading EVENT data: {data_type} {segment} for toxin: {toxin} ===")
    print(f"Trials after filtering: {list(toxin_trials)}")
    
    all_events = []
    
    for trial_string in toxin_trials:
        # Try different possible event file patterns
        possible_patterns = [
            f"events_{data_type}_{segment}_{trial_string}_simple_QC_final.csv",
            f"events_{data_type}_{toxin}_{trial_string}_{segment}_simple_QC_final.csv",
            f"events_{data_type}_{segment}_{toxin}_{trial_string}_simple_QC_final.csv"
        ]
        
        trial_dir = results_pipeline_dir / trial_string
        
        event_file_found = False
        for pattern in possible_patterns:
            file_path = trial_dir / pattern
            if file_path.exists():
                try:
                    event_data = pd.read_csv(file_path)
                    # Remove duplicates
                    event_data = event_data.drop_duplicates()
                    all_events.append(event_data)
                    event_file_found = True
                    print(f"✓ {trial_string}: Successfully loaded")
                    break
                except Exception as e:
                    print(f"✗ {trial_string}: Error loading - {e}")
        
        if not event_file_found:
            print(f"✗ {trial_string}: No event file found")
    
    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)
        return combined_events
    else:
        return None

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

'''
def plot_event_rate_comparison(event_data, toxin, save_path=None, stat_test='bootstrap'):
    """
    Create swarm plots comparing pre vs post event rates
    
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
            # Create swarm plot
            sns.swarmplot(data=type_data, x='segment', y='event_rate_per_100s', ax=ax, 
                         size=10, order=['pre', 'post'])
            
            # Remove top and right spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Set spine width to 2pt
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
            if toxin == '4AP':
                ax.set_xticklabels(['Pre', f'With 5mM 4AP'])
            if toxin == 'Ca_free':
                ax.set_xticklabels(['Pre', r'With Ca$^{2+}$ Free External'])
            
            # Statistical comparison
            pre_data = type_data[type_data['segment'] == 'pre']['event_rate_per_100s'].values
            post_data = type_data[type_data['segment'] == 'post']['event_rate_per_100s'].values
            
            if len(pre_data) > 0 and len(post_data) > 0:
                pairs = [('pre', 'post')]
                
                if stat_test == 'bootstrap':
                    p_value = statsf.bootstrap_test(pre_data, post_data)[0]
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
'''

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
            # Create half violin plot (split=True creates half violins)
            # Use inner=None to remove internal markings
            violin_parts = sns.violinplot(
                data=type_data, 
                x='segment', 
                y='event_rate_per_100s', 
                ax=ax,
                order=['pre', 'post'],
                inner=None,
                cut=0,
                linewidth=1.5,
                saturation=0.5,
                alpha=0.4
            )
            
            # Manually adjust violin positions to be half-width and offset
            for collection in ax.collections:
                # Get the paths of the violin plot
                if hasattr(collection, 'get_paths'):
                    paths = collection.get_paths()
                    for path in paths:
                        vertices = path.vertices
                        # Get the center x position
                        x_center = vertices[:, 0].mean()
                        # Only keep the right half of the violin
                        mask = vertices[:, 0] >= x_center
                        vertices[~mask, 0] = x_center
            
            # Overlay swarm plot with smaller dots
            sns.swarmplot(
                data=type_data, 
                x='segment', 
                y='event_rate_per_100s', 
                ax=ax,
                size=10,
                order=['pre', 'post'],
                #color='black',
                alpha=0.7,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Remove top and right spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Set spine width to 2pt
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
            if toxin == '4AP':
                ax.set_xticklabels(['Pre', f'With 5 mM 4-AP'])
            if toxin == 'Ca_free':
                ax.set_xticklabels(['Pre', r'With Ca$^{2+}$ Free External'])
            if toxin == 'ATP_1mM':
                ax.set_xticklabels(['Pre', f'With 1 mM ATP'])
            
            # Statistical comparison
            pre_data = type_data[type_data['segment'] == 'pre']['event_rate_per_100s'].values
            post_data = type_data[type_data['segment'] == 'post']['event_rate_per_100s'].values
            
            if len(pre_data) > 0 and len(post_data) > 0:
                pairs = [('pre', 'post')]
                
                if stat_test == 'bootstrap':
                    p_value = statsf.bootstrap_test(pre_data, post_data)[0]
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

'''
def analyze_toxin(toxin, save_dir=None):
    """
    Complete analysis for a specific toxin (both std and event rate)
    
    Parameters:
    -----------
    toxin : str
        Toxin name to analyze
    save_dir : str or Path, optional
        Directory to save figures
    """
    print(f"Analyzing toxin: {toxin}")
    print("="*50)

    # Calculate standard deviation data
    print("Loading standard deviation data...")
    std_data = calculate_std_for_toxin(toxin)

    # Calculate event rate data
    print("Loading event rate data...")
    event_data = calculate_event_rates_for_toxin(toxin)
    
    # Create plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        std_save_path = save_dir / f'{toxin}_std_comparison.png'
        event_save_path = save_dir / f'{toxin}_event_rate_comparison.png'
    else:
        std_save_path = None
        event_save_path = None
    
    # Plot standard deviation comparison
    if len(std_data) > 0:
        plot_std_comparison(std_data, toxin, std_save_path)
    else:
        print(f"No standard deviation data found for {toxin}")
    
    # Plot event rate comparison
    if len(event_data) > 0:
        plot_event_rate_comparison(event_data, toxin, event_save_path)
    else:
        print(f"No event rate data found for {toxin}")
    
    return std_data, event_data
'''
def analyze_toxin(toxin, save_dir=None, plot_std=True, plot_event_rate=True):
    """
    Complete analysis for a specific toxin
    
    Parameters:
    -----------
    toxin : str
        Toxin name to analyze
    save_dir : str or Path, optional
        Directory to save figures
    plot_std : bool
        Whether to calculate and plot standard deviation
    plot_event_rate : bool
        Whether to calculate and plot event rates
    """
    print(f"Analyzing toxin: {toxin}")
    print("="*50)

    std_data = None
    event_data = None
    
    # Calculate standard deviation data only if requested
    if plot_std:
        print("Loading standard deviation data...")
        std_data = calculate_std_for_toxin(toxin)

    # Calculate event rate data only if requested
    if plot_event_rate:
        print("Loading event rate data...")
        event_data = calculate_event_rates_for_toxin(toxin)
    
    # Create plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        std_save_path = save_dir / f'{toxin}_std_comparison.png'
        event_save_path = save_dir / f'{toxin}_event_rate_comparison.png'
    else:
        std_save_path = None
        event_save_path = None
    
    # Plot standard deviation comparison
    if plot_std and std_data is not None and len(std_data) > 0:
        plot_std_comparison(std_data, toxin, std_save_path)
    elif plot_std:
        print(f"No standard deviation data found for {toxin}")
    
    # Plot event rate comparison
    if plot_event_rate and event_data is not None and len(event_data) > 0:
        plot_event_rate_comparison(event_data, toxin, event_save_path)
    elif plot_event_rate:
        print(f"No event rate data found for {toxin}")
    
    return std_data, event_data

# Example usage
if __name__ == "__main__":
    # Define toxins to analyze
    toxins = ['Ani9_10uM', 'L-15_control', 'ATP_1mM', 'TRAM-34_1uM', 'dantrolene_10uM', 'DMSO_0.1%_control']
    toxins = ['4AP','dantrolene_10uM','Thapsigargin_1uM','Ca_free']
    toxins = ['TRAM-34_1uM']
    
    # Create save directory
    save_dir = data_dir / 'toxin_analysis_plots'
    
    # Analyze each toxin
    all_results = {}
    for toxin in toxins:
        try:
            std_data, event_data = analyze_toxin(toxin, save_dir, plot_std=False)
            all_results[toxin] = {'std_data': std_data, 'event_data': event_data}
        except Exception as e:
            print(f"Error analyzing {toxin}: {e}")
            continue
    
    print("Analysis complete!")