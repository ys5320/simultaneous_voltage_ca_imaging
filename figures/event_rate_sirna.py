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

def load_timeseries_data_by_expt(expt, data_type='voltage', segment='pre'):
    """
    Load and concatenate timeseries data for a specific experiment type, data type, and segment
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n']
    
    # Filter by expt
    expt_trials = df[df['expt'] == expt]['trial_string'].unique()
    
    # Debug: Print filtered trials
    print(f"\n=== Loading {data_type} {segment} for expt: {expt} ===")
    print(f"Trials after filtering: {list(expt_trials)}")
    
    all_data = []
    successful_trials = []
    
    for trial_string in expt_trials:
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

def load_event_data_by_expt(expt, data_type='voltage', segment='pre'):
    """
    Load and process event data for a specific experiment type, data type, and segment
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n']
    
    # Filter by expt
    expt_trials = df[df['expt'] == expt]['trial_string']
    
    # Debug: Print filtered trials
    print(f"\n=== Loading EVENT data: {data_type} {segment} for expt: {expt} ===")
    print(f"Trials after filtering: {list(expt_trials)}")
    
    all_events = []
    
    for trial_string in expt_trials:
        # Try different possible event file patterns
        possible_patterns = [
            f"events_{data_type}_{segment}_{trial_string}_simple_QC_final.csv",
            f"events_{data_type}_{expt}_{trial_string}_{segment}_simple_QC_final.csv",
            f"events_{data_type}_{segment}_{expt}_{trial_string}_simple_QC_final.csv"
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

def calculate_event_rates_for_comparison(expt_control, expt_treatment):
    """
    Calculate event rates for two experimental conditions (control vs treatment)
    
    Parameters:
    -----------
    expt_control : str
        Control experiment name (e.g., 'siRNA_negative_control')
    expt_treatment : str
        Treatment experiment name (e.g., 'siRNA_kcnn4')
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with event rates for plotting
    """
    results = []
    
    for expt, condition in [(expt_control, 'control'), (expt_treatment, 'treatment')]:
        for data_type in ['voltage', 'calcium']:
            for segment in ['pre', 'post']:
                # Load event data
                events = load_event_data_by_expt(expt, data_type, segment)
                
                # Filter events: only keep events with duration > 5s and < 50s
                if events is not None:
                    events = events[events['duration_sec'] > 5]
                    events = events[events['duration_sec'] < 50]
                
                # Load timeseries data to get total cell count and frame count
                timeseries_data, trials = load_timeseries_data_by_expt(expt, data_type, segment)
                
                if timeseries_data is not None:
                    # Get total frames (5 Hz sampling rate)
                    timeseries_cols = [col for col in timeseries_data.columns if col not in ['trial_string', 'cell_index']]
                    total_frames = len(timeseries_cols)
                    total_time_sec = total_frames / 5.0  # 5 Hz sampling
                    
                    # Iterate through ALL cells in timeseries
                    for idx, row in timeseries_data.iterrows():
                        trial = row['trial_string']
                        cell_idx = row['cell_index']
                        
                        event_count = 0
                        
                        if events is not None and len(events) > 0:
                            # Count positive events for this cell in this trial
                            cell_events = events[
                                (events['trial_string'] == trial) & 
                                (events['cell_index'] == cell_idx) & 
                                (events['event_type'] == 'positive')
                            ]
                            event_count = len(cell_events)
                        
                        # Calculate event rate per 100 seconds
                        event_rate_per_100s = (event_count / total_time_sec) * 100
                        
                        results.append({
                            'expt': expt,
                            'condition': condition,
                            'data_type': data_type,
                            'segment': segment,
                            'event_rate_per_100s': event_rate_per_100s,
                            'trial_string': trial,
                            'cell_index': cell_idx,
                            'total_events': event_count,
                            'total_time_sec': total_time_sec
                        })
    
    return pd.DataFrame(results)

def plot_siRNA_event_rate_comparison(event_data, save_path=None, stat_test='bootstrap'):
    """
    Create half violin + swarm plots comparing control vs treatment event rates
    
    Parameters:
    -----------
    event_data : pandas.DataFrame
        DataFrame with event rates
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
        type_data = event_data[event_data['data_type'] == data_type].copy()
        
        if len(type_data) > 0:
            # Create half violin plot
            violin_parts = sns.violinplot(
                data=type_data, 
                x='condition', 
                y='event_rate_per_100s', 
                ax=ax,
                order=['control', 'treatment'],
                inner=None,
                cut=0,
                linewidth=1.5,
                saturation=0.5,
                alpha=0.4
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
                x='condition', 
                y='event_rate_per_100s', 
                ax=ax,
                size=10,
                order=['control', 'treatment'],
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
            
            # Set x-tick labels
            ax.set_xticklabels(['Negative Control', 'KCNN4 Knockdown'])
            
            # Statistical comparison
            control_data = type_data[type_data['condition'] == 'control']['event_rate_per_100s'].values
            treatment_data = type_data[type_data['condition'] == 'treatment']['event_rate_per_100s'].values
            
            if len(control_data) > 0 and len(treatment_data) > 0:
                pairs = [('control', 'treatment')]
                
                if stat_test == 'bootstrap':
                    p_value = statsf.bootstrap_test_2sided(control_data, treatment_data)[0]
                    pvalues = [p_value]
                    annotator = Annotator(ax, pairs, data=type_data, x='condition', y='event_rate_per_100s')
                    annotator.configure(text_format='simple')
                    annotator.set_pvalues(pvalues).annotate()
                    
                elif stat_test == 'mann-whitney':
                    annotator = Annotator(ax, pairs, data=type_data, x='condition', y='event_rate_per_100s')
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

def save_siRNA_event_rate_info(event_data, save_dir):
    """
    Save trial counts, cell counts, mean event rates, and p-values to a text file
    
    Parameters:
    -----------
    event_data : pandas.DataFrame
        DataFrame with event rates
    save_dir : Path
        Directory to save the info file
    """
    info_path = save_dir / 'events_info_siRNA_comparison.txt'
    
    with open(info_path, 'w') as f:
        f.write(f"Event Rate Analysis: siRNA Negative Control vs KCNN4 Knockdown\n")
        f.write("="*60 + "\n\n")
        
        for data_type in ['voltage', 'calcium']:
            f.write(f"{data_type.upper()}\n")
            f.write("-"*60 + "\n")
            
            type_data = event_data[event_data['data_type'] == data_type]
            
            for condition in ['control', 'treatment']:
                condition_data = type_data[type_data['condition'] == condition]
                
                if len(condition_data) > 0:
                    n_cells = len(condition_data)
                    n_trials = condition_data['trial_string'].nunique()
                    mean_rate = condition_data['event_rate_per_100s'].mean()
                    std_rate = condition_data['event_rate_per_100s'].std()
                    n_zero_events = len(condition_data[condition_data['total_events'] == 0])
                    
                    # Extract slip and area counts
                    trial_strings = condition_data['trial_string'].unique()
                    
                    # Get unique slips (first 2 subparts)
                    slips = set()
                    for trial in trial_strings:
                        parts = trial.split('_')
                        if len(parts) >= 2:
                            slip_id = '_'.join(parts[:2])
                            slips.add(slip_id)
                    n_slips = len(slips)
                    
                    # Get unique areas (first 3 subparts)
                    areas = set()
                    for trial in trial_strings:
                        parts = trial.split('_')
                        if len(parts) >= 3:
                            area_id = '_'.join(parts[:3])
                            areas.add(area_id)
                    n_areas = len(areas)
                    
                    condition_name = 'NEGATIVE CONTROL' if condition == 'control' else 'KCNN4 KNOCKDOWN'
                    f.write(f"{condition_name}:\n")
                    f.write(f"  Number of trials: {n_trials}\n")
                    f.write(f"  Number of slips: {n_slips}\n")
                    f.write(f"  Number of areas: {n_areas}\n")
                    f.write(f"  Number of cells: {n_cells}\n")
                    f.write(f"  Cells with zero events: {n_zero_events}\n")
                    f.write(f"  Mean event rate: {mean_rate:.3f} ± {std_rate:.3f} events/100s\n")
                    
                    # List trials
                    trial_list = sorted(condition_data['trial_string'].unique())
                    f.write(f"  Trials: {', '.join(trial_list)}\n\n")
                else:
                    condition_name = 'NEGATIVE CONTROL' if condition == 'control' else 'KCNN4 KNOCKDOWN'
                    f.write(f"{condition_name}:\n")
                    f.write(f"  No data available\n\n")
            
            # Calculate p-value
            control_data = type_data[type_data['condition'] == 'control']['event_rate_per_100s'].values
            treatment_data = type_data[type_data['condition'] == 'treatment']['event_rate_per_100s'].values
            
            if len(control_data) > 0 and len(treatment_data) > 0:
                p_value = statsf.bootstrap_test_2sided(control_data, treatment_data)[0]
                f.write(f"  P-value (bootstrap, 2-sided): {p_value:.6f}\n")
                
                if len(control_data) != len(treatment_data):
                    f.write(f"  Note: Unpaired comparison (n_control={len(control_data)}, n_treatment={len(treatment_data)})\n")
            else:
                f.write(f"  P-value: Not enough data for comparison\n")
            
            f.write("\n")
    
    print(f"Event info saved to: {info_path}")

def analyze_siRNA_comparison(expt_control='siRNA_negative_control', 
                             expt_treatment='siRNA_kcnn4', 
                             save_dir=None):
    """
    Complete analysis comparing siRNA control vs treatment
    
    Parameters:
    -----------
    expt_control : str
        Control experiment name
    expt_treatment : str
        Treatment experiment name
    save_dir : str or Path, optional
        Directory to save figures
    """
    print(f"Comparing: {expt_control} vs {expt_treatment}")
    print("="*50)

    print("Loading event rate data...")
    event_data = calculate_event_rates_for_comparison(expt_control, expt_treatment)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        event_save_path = save_dir / 'siRNA_event_rate_comparison.png'
    else:
        event_save_path = None
    
    if event_data is not None and len(event_data) > 0:
        plot_siRNA_event_rate_comparison(event_data, event_save_path)
        
        if save_dir:
            save_siRNA_event_rate_info(event_data, save_dir)
    else:
        print(f"No event rate data found")
    
    return event_data

# Example usage
if __name__ == "__main__":
    # Create save directory
    save_dir = data_dir / 'siRNA_analysis_plots'
    
    # Analyze siRNA comparison
    try:
        event_data = analyze_siRNA_comparison(
            expt_control='siRNA_negative_control',
            expt_treatment='siRNA_kcnn4',
            save_dir=save_dir
        )
    except Exception as e:
        print(f"Error in analysis: {e}")
    
    print("Analysis complete!")