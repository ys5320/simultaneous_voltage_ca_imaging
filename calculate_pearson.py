from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

# Set up paths
home = Path.home()
if 'ys5320' in str(home):
    data_dir = Path(home, 'firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')
else:
    data_dir = Path('R:/home/firefly_link/Calcium_Voltage_Imaging/MDA_MB_468/analysis')

results_pipeline_dir = data_dir / 'results_profiles'

# Store all correlation coefficients
all_correlations = []

# Iterate through all folders
for folder in results_pipeline_dir.iterdir():
    if not folder.is_dir():
        continue
    
    folder_name = folder.name
    
    # Find voltage and calcium files
    voltage_files = list(folder.glob(f'pre_voltage_*_{folder_name}.csv'))
    calcium_files = list(folder.glob(f'pre_calcium_*_{folder_name}.csv'))
    
    # Check for mismatched pairs
    if len(voltage_files) != len(calcium_files):
        if len(voltage_files) > 0 or len(calcium_files) > 0:
            warnings.warn(f"Mismatched pair in {folder_name}: {len(voltage_files)} voltage, {len(calcium_files)} calcium files")
        continue
    
    # Skip if both missing
    if len(voltage_files) == 0:
        continue
    
    # Process each pair
    for v_file, c_file in zip(voltage_files, calcium_files):
        df_voltage = pd.read_csv(v_file, index_col=0)
        df_calcium = pd.read_csv(c_file, index_col=0)
        
        # Check same number of cells
        if df_voltage.shape[0] != df_calcium.shape[0]:
            warnings.warn(f"Different number of cells in {folder_name}: voltage={df_voltage.shape[0]}, calcium={df_calcium.shape[0]}")
            break
        
        # Calculate correlation for each cell
        for i in range(df_voltage.shape[0]):
            voltage_trace = -df_voltage.iloc[i].values #flip voltage values
            calcium_trace = df_calcium.iloc[i].values
            
            # Skip if NaN present
            if np.isnan(voltage_trace).any() or np.isnan(calcium_trace).any():
                continue
            
            corr, _ = pearsonr(voltage_trace, calcium_trace)
            all_correlations.append(corr)

# Calculate statistics
all_correlations = np.array(all_correlations)
mean_corr = np.mean(all_correlations)
std_corr = np.std(all_correlations)
median_corr = np.median(all_correlations)
q25 = np.percentile(all_correlations, 25)
q75 = np.percentile(all_correlations, 75)
iqr = q75 - q25

# Save results
output_file = results_pipeline_dir / 'Pearson_correlation_data.txt'
with open(output_file, 'w') as f:
    f.write(f"Pearson Correlation Analysis\n")
    f.write(f"{'='*40}\n")
    f.write(f"Total cells analyzed: {len(all_correlations)}\n")
    f.write(f"Mean ± SD: {mean_corr:.4f} ± {std_corr:.4f}\n") 
    f.write(f"Median: {median_corr:.4f}\n")
    f.write(f"IQR: {iqr:.4f}\n")
    f.write(f"Q25: {q25:.4f}\n")
    f.write(f"Q75: {q75:.4f}\n")

print(f"Analyzed {len(all_correlations)} cells")
print(f"Median correlation: {median_corr:.4f}")
print(f"IQR: {iqr:.4f}")
print(f"Results saved to: {output_file}")