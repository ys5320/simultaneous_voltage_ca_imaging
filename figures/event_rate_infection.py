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

def load_timeseries_data_by_expt(expt, data_type='voltage', segment='post'):
    """
    Load and concatenate timeseries data for a specific experiment type, data type, and segment
    Note: For 'condition' experiments, only 'post' segment exists
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n_focus']
    
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

def load_event_data_by_expt(expt, data_type='voltage', segment='post'):
    """
    Load and process event data for a specific experiment type, data type, and segment
    Note: For 'condition' experiments, only 'post' segment exists
    """
    # Load the main dataframe
    df = pd.read_csv(df_path)
    df = df[df['use'] != 'n_focus']
    
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
                    print(f"✓ {trial_string}: Successfully loaded from {pattern}")
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

def calculate_event_rates_for_conditions(conditions=['condition_o', 'condition_e', 'condition_c', 'condition_d']):
    """
    Calculate event rates for condition experiments
    
    Parameters:
    -----------
    conditions : list
        List of condition experiment names
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with event rates for plotting
    """
    results = []
    
    for expt in conditions:
        # Only voltage and only post segment for condition experiments
        data_type = 'voltage'
        segment = 'post'
        
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
                    'condition': expt,
                    'event_rate_per_100s': event_rate_per_100s,
                    'trial_string': trial,
                    'cell_index': cell_idx,
                    'total_events': event_count,
                    'total_time_sec': total_time_sec
                })
    
    return pd.DataFrame(results)

def plot_condition_event_rates(event_data, save_path=None):
    """
    Create violin + swarm plot comparing event rates across four conditions
    
    Parameters:
    -----------
    event_data : pandas.DataFrame
        DataFrame with event rates
    save_path : str or Path, optional
        Path to save the figure
    """
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Ensure correct order
    condition_order = ['condition_o', 'condition_e', 'condition_c', 'condition_d']
    
    # Create violin plot
    violin_parts = ax.violinplot(
        [event_data[event_data['condition'] == cond]['event_rate_per_100s'].values 
         for cond in condition_order],
        positions=range(len(condition_order)),
        showmeans=False,
        showmedians=False,
        widths=0.7
    )
    
    # Color the violins
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.4)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Overlay swarm plot
    for i, (cond, color) in enumerate(zip(condition_order, colors)):
        cond_data = event_data[event_data['condition'] == cond]['event_rate_per_100s'].values
        x = np.random.normal(i, 0.04, size=len(cond_data))  # Add jitter
        ax.scatter(x, cond_data, s=100, alpha=0.7, color=color, 
                  edgecolor='white', linewidth=0.5, zorder=3)
    
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Set spine width to 2pt
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Set labels
    ax.set_title('Voltage Hyperpolarization Event Rate\nAcross Experimental Conditions', fontsize=22, pad=20)
    ax.set_ylabel('Events per 100s', fontsize=20)
    ax.set_xlabel('Condition', fontsize=20)
    
    # Set x-tick labels (you can customize these labels)
    ax.set_xticks(range(len(condition_order)))
    ax.set_xticklabels(['Control\n(condition_o)', 'Condition E', 'Condition C', 'Condition D'], 
                       fontsize=18)
    
    # Add statistical comparisons using bootstrap test
    # Compare each condition against control (condition_o)
    control_data = event_data[event_data['condition'] == 'condition_o']['event_rate_per_100s'].values
    
    # Prepare pairs for statistical annotation
    pairs = [(0, i) for i in range(1, len(condition_order))]
    pvalues = []
    
    for i in range(1, len(condition_order)):
        treatment_data = event_data[event_data['condition'] == condition_order[i]]['event_rate_per_100s'].values
        if len(control_data) > 0 and len(treatment_data) > 0:
            p_value = statsf.bootstrap_test_2sided(control_data, treatment_data)[0]
            pvalues.append(p_value)
        else:
            pvalues.append(1.0)
    
    # Add significance bars
    y_max = event_data['event_rate_per_100s'].max()
    y_range = event_data['event_rate_per_100s'].max() - event_data['event_rate_per_100s'].min()
    
    for idx, (pair, pval) in enumerate(zip(pairs, pvalues)):
        y_pos = y_max + y_range * (0.1 + idx * 0.15)
        ax.plot([pair[0], pair[1]], [y_pos, y_pos], 'k-', linewidth=1.5)
        
        # Format p-value
        if pval < 0.001:
            sig_text = '***'
        elif pval < 0.01:
            sig_text = '**'
        elif pval < 0.05:
            sig_text = '*'
        else:
            sig_text = f'p={pval:.3f}'
        
        ax.text((pair[0] + pair[1]) / 2, y_pos, sig_text, 
               ha='center', va='bottom', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        eps_path = str(save_path).replace('.png', '.eps')
        plt.savefig(eps_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
        print(f"EPS saved to: {eps_path}")
    
    plt.show()

def save_condition_event_info(event_data, save_dir):
    """
    Save trial counts, cell counts, mean event rates, and p-values to a text file
    
    Parameters:
    -----------
    event_data : pandas.DataFrame
        DataFrame with event rates
    save_dir : Path
        Directory to save the info file
    """
    info_path = save_dir / 'events_info_conditions_comparison.txt'
    
    condition_order = ['condition_o', 'condition_e', 'condition_c', 'condition_d']
    
    with open(info_path, 'w') as f:
        f.write(f"Voltage Event Rate Analysis: Condition Experiments\n")
        f.write("="*60 + "\n\n")
        
        for condition in condition_order:
            f.write(f"{condition.upper()}\n")
            f.write("-"*60 + "\n")
            
            condition_data = event_data[event_data['condition'] == condition]
            
            if len(condition_data) > 0:
                n_cells = len(condition_data)
                n_trials = condition_data['trial_string'].nunique()
                mean_rate = condition_data['event_rate_per_100s'].mean()
                std_rate = condition_data['event_rate_per_100s'].std()
                median_rate = condition_data['event_rate_per_100s'].median()
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
                
                f.write(f"  Number of trials: {n_trials}\n")
                f.write(f"  Number of slips: {n_slips}\n")
                f.write(f"  Number of areas: {n_areas}\n")
                f.write(f"  Number of cells: {n_cells}\n")
                f.write(f"  Cells with zero events: {n_zero_events}\n")
                f.write(f"  Mean event rate: {mean_rate:.3f} ± {std_rate:.3f} events/100s\n")
                f.write(f"  Median event rate: {median_rate:.3f} events/100s\n")
                
                # List trials
                trial_list = sorted(condition_data['trial_string'].unique())
                f.write(f"  Trials: {', '.join(trial_list)}\n\n")
            else:
                f.write(f"  No data available\n\n")
        
        # Statistical comparisons against control (condition_o)
        f.write("\nSTATISTICAL COMPARISONS (vs. condition_o control)\n")
        f.write("="*60 + "\n")
        
        control_data = event_data[event_data['condition'] == 'condition_o']['event_rate_per_100s'].values
        
        for condition in ['condition_e', 'condition_c', 'condition_d']:
            treatment_data = event_data[event_data['condition'] == condition]['event_rate_per_100s'].values
            
            if len(control_data) > 0 and len(treatment_data) > 0:
                p_value = statsf.bootstrap_test_2sided(control_data, treatment_data)[0]
                f.write(f"\n{condition} vs condition_o:\n")
                f.write(f"  P-value (bootstrap, 2-sided): {p_value:.6f}\n")
                f.write(f"  n_control={len(control_data)}, n_treatment={len(treatment_data)}\n")
            else:
                f.write(f"\n{condition} vs condition_o:\n")
                f.write(f"  P-value: Not enough data for comparison\n")
    
    print(f"Event info saved to: {info_path}")

def analyze_conditions(conditions=['condition_o', 'condition_e', 'condition_c', 'condition_d'],
                       save_dir=None):
    """
    Complete analysis comparing voltage event rates across conditions
    
    Parameters:
    -----------
    conditions : list
        List of condition experiment names
    save_dir : str or Path, optional
        Directory to save figures
    """
    print(f"Analyzing conditions: {conditions}")
    print("="*50)

    print("Loading event rate data...")
    event_data = calculate_event_rates_for_conditions(conditions)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        event_save_path = save_dir / 'conditions_voltage_event_rate_comparison.png'
    else:
        event_save_path = None
    
    if event_data is not None and len(event_data) > 0:
        print(f"\nTotal cells across all conditions: {len(event_data)}")
        print("\nCells per condition:")
        for cond in conditions:
            n_cells = len(event_data[event_data['condition'] == cond])
            print(f"  {cond}: {n_cells} cells")
        
        plot_condition_event_rates(event_data, event_save_path)
        
        if save_dir:
            save_condition_event_info(event_data, save_dir)
    else:
        print(f"No event rate data found")
    
    return event_data

# Example usage
if __name__ == "__main__":
    # Create save directory
    save_dir = data_dir / 'condition_analysis_plots'
    
    # Analyze conditions
    try:
        event_data = analyze_conditions(
            conditions=['condition_o', 'condition_e', 'condition_c', 'condition_d'],
            save_dir=save_dir
        )
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("Analysis complete!")