import pandas as pd
from pathlib import Path

def combine_existing_qc_finals(base_results_dir, df_metadata):
    """Combine existing QC_final files by experiment type without re-doing QC"""
    
    # Create mapping from trial_string to experiment type
    trial_to_expt = {}
    for idx, row in df_metadata.iterrows():
        trial_to_expt[row['trial_string']] = row['expt']
    
    # Group trials by experiment type
    expt_to_trials = {}
    for trial_string, expt in trial_to_expt.items():
        if expt not in expt_to_trials:
            expt_to_trials[expt] = []
        expt_to_trials[expt].append(trial_string)
    
    print(f"Found experiment types: {list(expt_to_trials.keys())}")
    
    base_path = Path(base_results_dir)
    
    # Create events_df directory if it doesn't exist
    events_df_dir = base_path / 'events_df'
    events_df_dir.mkdir(exist_ok=True)
    
    # For each experiment type
    for expt, trial_list in expt_to_trials.items():
        print(f"\nProcessing experiment: {expt}")
        print(f"Trials: {trial_list}")
        
        # For each data_type and segment combination
        for data_type in ['voltage', 'calcium']:
            for segment in ['pre', 'post']:
                all_events_list = []
                
                for trial_string in trial_list:
                    trial_dir = base_path / trial_string
                    
                    # Look for QC_final file with the new naming convention
                    qc_file = trial_dir / f"events_{data_type}_{expt}_{segment}_{trial_string}_simple_QC_final.csv"
                    
                    if qc_file.exists():
                        print(f"  Found: {qc_file.name}")
                        
                        try:
                            events_df = pd.read_csv(qc_file)
                            
                            # Add trial metadata
                            events_df['trial_string'] = trial_string
                            trial_row = df_metadata[df_metadata['trial_string'] == trial_string]
                            if len(trial_row) > 0:
                                trial_info = trial_row.iloc[0]
                                events_df['date'] = trial_info.get('date', '')
                                events_df['area'] = trial_info.get('area', '')
                                events_df['toxin'] = trial_info.get('expt', expt)
                                events_df['concentration'] = trial_info.get('concentration', '')
                            
                            all_events_list.append(events_df)
                            
                        except Exception as e:
                            print(f"  Error reading {qc_file.name}: {e}")
                    else:
                        print(f"  Missing: events_{data_type}_{segment}_{trial_string}_simple_QC_final.csv")
                
                # Combine and save if we have data
                if all_events_list:
                    combined_events = pd.concat(all_events_list, ignore_index=True)
                    
                    # Save combined file
                    summary_file = events_df_dir / f"events_{data_type}_{segment}_{expt}_QC_final.csv"
                    combined_events.to_csv(summary_file, index=False)
                    
                    print(f"  ✓ Created: {summary_file.name}")
                    print(f"    Total events: {len(combined_events)} from {len(all_events_list)} trials")
                else:
                    print(f"  - No data found for {data_type} {segment} in {expt}")
    
    print(f"\nSummary files created in: {events_df_dir}")


home = Path.home()
date = '20250827'
cell_line = 'MDA_MB_468'
if "ys5320" in str(home):
    top_dir = Path(home, "firefly_link/Calcium_Voltage_Imaging",f'{cell_line}')
else:
    top_dir = Path(r'R:\home\firefly_link\Calcium_Voltage_Imaging\MDA_MB_468')
df = Path(top_dir,'analysis', 'dataframes',f'long_acqs_{cell_line}_all_before_{date}.csv')
base_results_dir = Path(top_dir,'analysis', 'results_pipeline')
# Usage:
toxins = ['siRNA']
df_data = pd.read_csv(df)
df_data = df_data[df_data['expt'].apply(lambda x: any(k in x for k in toxins))]

df_data = df_data.reset_index(drop=True)  

combine_existing_qc_finals(base_results_dir, df_data)