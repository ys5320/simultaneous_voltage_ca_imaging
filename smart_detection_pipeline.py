import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tifffile as tiff
from scipy.ndimage import gaussian_filter1d, binary_opening, binary_closing
# Import functions from infusion_detection.py
from infusion_detection import (
    ensure_numeric_data,
    apply_highpass_filter,
    detect_multiple_channels,
    find_consensus_timepoint,
    analyze_detection_consistency,
    detrend
)


def determine_toxin_from_trial(trial_string, filename):
    """Determine toxin type from trial string or filename"""
    filename_lower = filename.lower()
    trial_lower = trial_string.lower()
    
    if 'l-15' in filename_lower:
        return 'L-15_control'
    elif 'ani9' in filename_lower:
        return 'Ani9_10uM'
    elif 'atp' in filename_lower:
        return 'ATP_1mM'
    elif 'tram' in filename_lower:
        return 'TRAM-34_1uM'
    elif 'dantrolene' in filename_lower:
        return 'dantrolene_10uM'
    elif 'dmso' in filename_lower:
        return 'DMSO_0.1%_control'
    else:
        return 'unknown'  # Handle as needed

def apply_highpass_to_full_timeseries(data_matrix, sampling_rate_hz=5, cutoff_freq=0.01, apply_normalization=True, data_type="voltage"):
    """
    Step 1: Apply normalization first, then high-pass filter to the entire timeseries
    """
    print(f"=== STEP 1: Normalization + High-Pass Filter to Full {data_type.upper()} Timeseries ===")
    
    # Ensure data is numeric
    data_array = ensure_numeric_data(data_matrix)
    
    # Apply normalization FIRST (before any filtering)
    if apply_normalization:
        print(f"Applying normalization to raw {data_type} data (before filtering)...")
        normalized_data = np.zeros_like(data_array)
        for i in range(data_array.shape[0]):
            channel_data = data_array[i, :]
            
            # Min-max normalization to [-1, 1] range
            channel_min = np.min(channel_data)
            channel_max = np.max(channel_data)
            channel_range = channel_max - channel_min
            
            if channel_range > 0:  # Avoid division by zero
                # Normalize to [0, 1] then shift to [-1, 1]
                normalized_data[i, :] = 2 * (channel_data - channel_min) / channel_range - 1
            else:
                # If channel is completely flat, set to zero
                normalized_data[i, :] = 0
        
        data_array = normalized_data
        print(f"  - Min-max normalization applied to {data_type} (range: [-1, 1])")
    
    # Apply high-pass filter to normalized data
    print(f"Applying high-pass filter to normalized {data_type} data (cutoff: {cutoff_freq} Hz)...")
    filtered_data = np.zeros_like(data_array)
    for i in range(data_array.shape[0]):
        filtered_data[i, :] = apply_highpass_filter(data_array[i, :], sampling_rate_hz, cutoff_freq)
    
    print(f"Normalization + High-pass filtering complete for {data_type}. Shape: {filtered_data.shape}")
    return filtered_data

def find_consensus_timepoint_from_voltage(voltage_highpass_data, sampling_rate_hz=5, time_window=(4500, 5500), 
                                        detection_method='zscore', consensus_method='mode'):
    """
    Step 2: Find consensus timepoint using VOLTAGE high-pass filtered data ONLY
    """
    print("=== STEP 2: Finding Consensus Timepoint from VOLTAGE data ===")
    
    # Use high-pass filtered data directly for detection  
    detection_data = voltage_highpass_data.copy()
    
    # Run detection
    consensus_points, channel_results, window_bounds = detect_multiple_channels(
        detection_data,
        sampling_rate_hz=sampling_rate_hz,
        time_window=time_window,
        method=detection_method,
        consensus_threshold=0.5,
        threshold=2.5,
        single_detection=True,
        apply_filter=False  # Already filtered
    )
    
    # Analyze consistency
    print("\n=== Detection Consistency Analysis ===")
    analyze_detection_consistency(channel_results, sampling_rate_hz)
    
    # Find consensus timepoint
    print(f"\n=== Consensus Timepoint ({consensus_method.upper()}) ===")
    consensus_timepoint, final_results = find_consensus_timepoint(
        channel_results, sampling_rate_hz, method=consensus_method)
    
    return consensus_timepoint

def segment_highpass_timeseries(highpass_data, consensus_timepoint, sampling_rate_hz=5, data_type="voltage"):
    """
    Step 3: Segment the high-pass filtered timeseries using consensus timepoint
    """
    print(f"=== STEP 3: Segmenting High-Pass Filtered {data_type.upper()} Timeseries ===")
    
    n_channels, n_timepoints = highpass_data.shape
    
    # Define segment boundaries
    pre_start = 200
    pre_end = consensus_timepoint - 200
    post_start = consensus_timepoint + 400
    post_end = n_timepoints - 1
    
    # Validate and create segments
    segments = {}
    
    if pre_start >= pre_end:
        print(f"Warning: {data_type} Pre-segment invalid (start: {pre_start}, end: {pre_end})")
        segments['pre'] = None
    else:
        segments['pre'] = highpass_data[:, pre_start:pre_end+1]
    
    if post_start >= post_end:
        print(f"Warning: {data_type} Post-segment invalid (start: {post_start}, end: {post_end})")
        segments['post'] = None
    else:
        segments['post'] = highpass_data[:, post_start:post_end+1]
    
    # Print segment info
    samples_per_min = sampling_rate_hz * 60
    print(f"Consensus timepoint: {consensus_timepoint} frames ({consensus_timepoint/samples_per_min:.2f} min)")
    
    if segments['pre'] is not None:
        print(f"{data_type.upper()} PRE segment: frames {pre_start}-{pre_end} ({pre_start/samples_per_min:.2f}-{pre_end/samples_per_min:.2f} min)")
        print(f"  Shape: {segments['pre'].shape}")
    
    if segments['post'] is not None:
        print(f"{data_type.upper()} POST segment: frames {post_start}-{post_end} ({post_start/samples_per_min:.2f}-{post_end/samples_per_min:.2f} min)")
        print(f"  Shape: {segments['post'].shape}")
    
    return segments

def apply_segment_processing(segment_data, apply_mean_center=True, apply_detrend=True, 
                           apply_gaussian=False, gaussian_sigma=3, data_type="voltage"):
    """
    Step 4: Apply processing to individual segments (normalization already done)
    """
    if segment_data is None:
        return None, None
    
    # Start with normalized + high-pass filtered data
    processed_data = segment_data.copy()
    
    # Step 4a: Mean center each channel in this segment
    if apply_mean_center:
        mean_centered_data = np.zeros_like(processed_data)
        for i in range(processed_data.shape[0]):
            channel_mean = np.mean(processed_data[i, :])
            mean_centered_data[i, :] = processed_data[i, :] - channel_mean
        processed_data = mean_centered_data
        print(f"  - {data_type} mean centering applied")
    
    # Step 4b: Apply linear detrending
    if apply_detrend:
        detrended_data = np.zeros_like(processed_data)
        for i in range(processed_data.shape[0]):
            detrended_data[i, :] = detrend(processed_data[i, :], type='linear')
        processed_data = detrended_data
        print(f"  - {data_type} linear detrending applied")
    
    # Step 4c: Apply Gaussian filter
    if apply_gaussian:
        gaussian_data = np.zeros_like(processed_data)
        for i in range(processed_data.shape[0]):
            gaussian_data[i, :] = gaussian_filter1d(processed_data[i, :], sigma=gaussian_sigma)
        
        # Store non-gaussian data for plotting comparison
        non_gaussian_data = processed_data.copy()
        processed_data = gaussian_data
        print(f"  - {data_type} Gaussian filtering applied (sigma={gaussian_sigma})")
        
        # Return Gaussian filtered data as main, non-gaussian as secondary
        return processed_data, non_gaussian_data
    else:
        # No Gaussian filtering
        return processed_data, None

def detect_file_type_and_datapoints(df):
    """
    Detect if this is a full experiment or post-only file
    Returns: (file_type, n_datapoints)
    """
    # Find where data columns start (after 'cell_y')
    if 'cell_y' in df.columns:
        cell_y_idx = df.columns.get_loc('cell_y')
        data_columns = df.columns[cell_y_idx + 1:]
    else:
        # Fallback: assume data columns are numeric
        data_columns = [col for col in df.columns if str(col).isdigit()]
    
    n_datapoints = len(data_columns)
    
    # Determine file type based on datapoint count
    if n_datapoints >= 9000:  # Around 10,000
        file_type = "full_experiment"
    elif n_datapoints >= 4000:  # Around 5,000
        file_type = "post_only"
    else:
        file_type = "unknown"
    
    print(f"Detected: {file_type} with {n_datapoints} datapoints")
    return file_type, n_datapoints

def save_segment_data(segment_data, segment_name, toxin, trial_string, save_dir_data, 
                     cell_positions=None, original_filename=None, data_type="voltage"):
    """
    Save segment data to CSV with original cell IDs and positions preserved
    """
    if segment_data is None:
        return None
    
    if save_dir_data is not None:
        data_path = Path(save_dir_data) / f"{segment_name}_{data_type}_{toxin}_{trial_string}.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame from segment data with original cell IDs as index
        if cell_positions is not None and 'cell_id' in cell_positions.columns:
            # Use original cell IDs as DataFrame index
            df = pd.DataFrame(segment_data, index=cell_positions['cell_id'])
            
            # Add cell positions as columns
            df['cell_id'] = cell_positions['cell_id'].values
            df['cell_x'] = cell_positions['cell_x'].values
            df['cell_y'] = cell_positions['cell_y'].values
            
            # Reset index to make cell_Id a regular column, but keep the data aligned
            df = df.reset_index(drop=True)
        else:
            # Fallback: sequential indexing
            df = pd.DataFrame(segment_data)
            if cell_positions is not None:
                if 'cell_id' in cell_positions.columns:
                    df['cell_id'] = cell_positions['cell_id'].values
                df['cell_x'] = cell_positions['cell_x'].values
                df['cell_y'] = cell_positions['cell_y'].values
        
        df.to_csv(data_path, index=False)
        print(f"  {data_type.capitalize()} data saved with original cell IDs to: {data_path}")
        return data_path
    
    return None

def plot_raw_and_filtered_segment(raw_segment, processed_segment, non_gaussian_segment, 
                                segment_name, toxin, trial_string, sampling_rate_hz=5, 
                                save_dir_plots=None, max_channels_plot=10, data_type="voltage"):
    """
    Step 5b: Create plot showing raw and filtered data
    """
    if raw_segment is None:
        print(f"Cannot plot {data_type} {segment_name} segment - data is None")
        return None
    
    n_channels, n_timepoints = raw_segment.shape
    time_axis = np.arange(n_timepoints) / sampling_rate_hz
    
    # Limit channels for clarity
    n_channels_plot = n_channels
    
    fig, ax = plt.subplots(figsize=(10,20))
    
    # Calculate channel spacing
    data_ranges = [np.ptp(raw_segment[i, :]) for i in range(n_channels_plot)]
    max_range = max(data_ranges) if data_ranges else 1
    channel_spacing = max_range * 1.5
    
    for i in range(n_channels_plot):
        offset = i * channel_spacing
        
        # Plot raw data (grey, thinner line)
        ax.plot(time_axis, raw_segment[i, :] + offset, color='grey', 
               linewidth=0.8, alpha=0.6, label='Normalized + High-pass' if i == 0 else '')
        
        # Plot intermediate processing (mean-centered + detrended, black, if available)
        if non_gaussian_segment is not None:
            ax.plot(time_axis, non_gaussian_segment[i, :] + offset, color='black', 
                   linewidth=1.0, alpha=0.7, label='Mean-centered + Detrended' if i == 0 else '')
        
        # Plot final processed data (red, thicker line) - this is what's used for analysis
        if processed_segment is not None:
            is_gaussian = non_gaussian_segment is not None  # If we have intermediate, then processed is Gaussian
            label = 'Final (Gaussian filtered)' if is_gaussian and i == 0 else 'Final (processed)' if i == 0 else ''
            color = 'red' if is_gaussian else 'black'
            linewidth = 1.2 if is_gaussian else 1.0
            
            ax.plot(time_axis, processed_segment[i, :] + offset, color=color, 
                   linewidth=linewidth, alpha=0.9, label=label)
        
        # Add channel label
        ax.text(-0.02 * np.max(time_axis), np.mean(raw_segment[i, :]) + offset, f'{i}', 
                verticalalignment='center', fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    
    # Update title to reflect what's being shown
    gaussian_text = " (with Gaussian)" if non_gaussian_segment is not None else ""
    ax.set_title(f'{segment_name.upper()} Segment - {data_type.upper()} - {toxin} {trial_string}\n(Processing Pipeline{gaussian_text})', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, np.max(time_axis))
    ax.set_ylim(-channel_spacing*0.5, (n_channels_plot-0.5) * channel_spacing)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    if n_channels_plot > 0:
        ax.legend(loc='upper right', fontsize=10)
    
    # Add text box showing what data is used for analysis
    analysis_text = "Analysis uses: Gaussian filtered data" if non_gaussian_segment is not None else "Analysis uses: Mean-centered + detrended data"
    ax.text(0.02, 0.02, analysis_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    if save_dir_plots is not None:
        plot_path = Path(save_dir_plots) / f"{segment_name}_{data_type}_{toxin}_{trial_string}_processing.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  {data_type.capitalize()} plot saved to: {plot_path}")
    
    plt.close()  # Close to save memory
    return plot_path

def detect_voltage_events_enhanced_threshold(data_segment, sampling_rate_hz=5, threshold_multiplier=2.5, 
                                            apply_gaussian_for_detection=False, min_event_duration_sec=1.0, 
                                            use_global_threshold=True, use_fixed_std=False, fixed_std_value=None,
                                            cell_id_mapping=None, data_type="voltage", **kwargs):
    """
    Enhanced event detection with multiple threshold strategies:
    1. Global threshold + per-channel baselines (original)
    2. Per-channel thresholds + per-channel baselines 
    3. Fixed STD + per-channel baselines (NEW)
    """
    n_channels, n_timepoints = data_segment.shape
    events_list = []
    
    min_event_samples = int(min_event_duration_sec * sampling_rate_hz)
    
    # Determine which threshold strategy to use
    if use_fixed_std:
        print(f"Using FIXED STD {data_type} event detection")
        threshold_strategy = "fixed_std"
    elif use_global_threshold:
        print(f"Using GLOBAL STD {data_type} event detection")
        threshold_strategy = "global_std"
    else:
        print(f"Using PER-CHANNEL STD {data_type} event detection")
        threshold_strategy = "per_channel_std"
    
    print(f"Min duration: {min_event_duration_sec}s, Threshold: {threshold_multiplier}σ")
    
    # Calculate thresholds based on strategy
    channel_baselines = []
    channel_thresholds = []
    
    if use_fixed_std:
        # FIXED STD APPROACH: Use a predetermined STD value for all channels
        if fixed_std_value is None:
            # If no fixed value provided, calculate from global data as reference
            all_data_flattened = data_segment.flatten()
            global_std = np.std(all_data_flattened)
            fixed_std_value = global_std
            print(f"  No fixed_std_value provided, using global std: {fixed_std_value:.4f}")
        else:
            print(f"  Using provided fixed std: {fixed_std_value:.4f}")
        
        fixed_threshold = threshold_multiplier * fixed_std_value
        print(f"  Fixed threshold for all channels: ±{fixed_threshold:.4f}")
        
        # Calculate per-channel baselines but use same threshold
        for i in range(n_channels):
            channel_data = data_segment[i, :]
            baseline = np.median(channel_data)  # Per-channel median baseline
            
            channel_baselines.append(baseline)
            channel_thresholds.append(fixed_threshold)  # Same threshold for all
        
        print(f"  Using per-channel median baselines with fixed STD threshold")
        
    elif use_global_threshold:
        # GLOBAL STD APPROACH: Calculate global stats, use per-channel baselines
        print(f"Calculating GLOBAL statistics across all {data_type} channels...")
        
        all_data_flattened = data_segment.flatten()
        global_mean = np.mean(all_data_flattened)
        global_std = np.std(all_data_flattened)
        global_threshold = threshold_multiplier * global_std
        
        print(f"  Global {data_type} mean: {global_mean:.4f}")
        print(f"  Global {data_type} std: {global_std:.4f}")
        print(f"  Global {data_type} threshold: ±{global_threshold:.4f}")
        
        # Calculate per-channel baselines but use global threshold
        for i in range(n_channels):
            channel_data = data_segment[i, :]
            baseline = np.median(channel_data)  # Per-channel median baseline
            
            channel_baselines.append(baseline)
            channel_thresholds.append(global_threshold)  # Same threshold for all
        
        print(f"  Using per-channel median baselines with global STD threshold")
        
    else:
        # PER-CHANNEL STD APPROACH: Each channel gets its own threshold
        print(f"Calculating PER-CHANNEL {data_type} statistics...")
        
        for i in range(n_channels):
            channel_data = data_segment[i, :]
            baseline = np.median(channel_data)
            channel_std = np.std(channel_data)
            channel_threshold = threshold_multiplier * channel_std
            
            channel_baselines.append(baseline)
            channel_thresholds.append(channel_threshold)
        
        print(f"  Using per-channel median baselines with per-channel STD thresholds")
    
    # Event detection loop (same for all strategies)
    print(f"Detecting {data_type} events using {threshold_strategy} approach...")
    
    for channel_idx in range(n_channels):
        channel_data = data_segment[channel_idx, :]
        baseline = channel_baselines[channel_idx]
        threshold = channel_thresholds[channel_idx]
        
        # Find threshold crossings
        deviation_from_baseline = np.abs(channel_data - baseline)
        threshold_mask = deviation_from_baseline > threshold
        
        # Apply morphological operations
        if data_type == "voltage":
            # Apply aggressive filtering for voltage (original behavior)
            print(f"  - Applying full morphological filtering for voltage channel {channel_idx}")
            for _ in range(2):
                threshold_mask = binary_opening(threshold_mask)
            for _ in range(2):
                threshold_mask = binary_closing(threshold_mask)
                
        elif data_type == "calcium":
            '''
            # Apply minimal filtering for calcium (just closing to fill small gaps)
            print(f"  - Applying minimal morphological filtering for calcium channel {channel_idx}")
            for _ in range(1):  # Only one round of closing
                threshold_mask = binary_closing(threshold_mask)
            # Skip binary_opening entirely for calcium
            '''
            # Apply aggressive filtering for voltage (original behavior)
            print(f"  - Applying full morphological filtering for voltage channel {channel_idx}")
            for _ in range(2):
                threshold_mask = binary_opening(threshold_mask)
            for _ in range(2):
                threshold_mask = binary_closing(threshold_mask)
            
        else:
            # Default: no morphological filtering for unknown data types
            print(f"  - No morphological filtering applied for {data_type} channel {channel_idx}")
        
        # Find continuous regions
        threshold_starts = np.where(np.diff(threshold_mask.astype(int)) == 1)[0] + 1
        threshold_ends = np.where(np.diff(threshold_mask.astype(int)) == -1)[0] + 1
        
        # Handle edge cases
        if threshold_mask[0]:
            threshold_starts = np.concatenate([[0], threshold_starts])
        if threshold_mask[-1]:
            threshold_ends = np.concatenate([threshold_ends, [len(threshold_mask)-1]])
        
        # Process each threshold-crossing event
        for thresh_start, thresh_end in zip(threshold_starts, threshold_ends):
            
            # Skip very short threshold crossings
            if thresh_end - thresh_start < min_event_samples // 2:
                continue
            
            # Determine event polarity
            event_region = channel_data[thresh_start:thresh_end+1]
            mean_deviation = np.mean(event_region - baseline)
            event_type = 'positive' if mean_deviation > 0 else 'negative'
            
            # Refine boundaries to baseline crossings
            event_start = thresh_start
            for i in range(thresh_start - 1, -1, -1):
                if event_type == 'positive':
                    if channel_data[i] <= baseline:
                        event_start = i + 1
                        break
                else:
                    if channel_data[i] >= baseline:
                        event_start = i + 1
                        break
                event_start = i
            
            event_end = thresh_end
            for i in range(thresh_end + 1, len(channel_data)):
                if event_type == 'positive':
                    if channel_data[i] <= baseline:
                        event_end = i - 1
                        break
                else:
                    if channel_data[i] >= baseline:
                        event_end = i - 1
                        break
                event_end = i
            
            # Apply minimum duration filter
            duration_samples = event_end - event_start + 1
            if duration_samples < min_event_samples:
                continue
            
            # Calculate event properties
            event_segment = channel_data[event_start:event_end+1]
            
            if event_type == 'positive':
                amplitude = np.max(event_segment) - baseline
                peak_value = np.max(event_segment)
            else:
                amplitude = baseline - np.min(event_segment)
                peak_value = np.min(event_segment)
            
            # Store event properties with strategy info
            # Map channel index to original cell ID
            original_cell_id = channel_idx  # Default fallback
            if cell_id_mapping is not None and len(cell_id_mapping) > channel_idx:
                original_cell_id = cell_id_mapping.iloc[channel_idx]['cell_id']

            # Store event properties with original cell ID
            event_properties = {
                'channel_idx': channel_idx,  # Keep for internal processing
                'original_cell_id': original_cell_id,  # NEW: Original cell identifier
                'start_time_sec': event_start / sampling_rate_hz,
                'end_time_sec': event_end / sampling_rate_hz,
                'start_sample': event_start,
                'end_sample': event_end,
                'duration_sec': duration_samples / sampling_rate_hz,
                'duration_samples': duration_samples,
                'amplitude': amplitude,
                'event_type': event_type,
                'mean_amplitude': np.abs(np.mean(event_segment - baseline)),
                'peak_value': peak_value,
                'baseline_value': baseline,
                'threshold_used': threshold,
                'threshold_type': threshold_strategy,
                'threshold_start_sample': thresh_start,
                'threshold_end_sample': thresh_end,
                'data_type': data_type
            }
            
            # Add strategy-specific metadata
            if use_fixed_std:
                event_properties['fixed_std_value'] = fixed_std_value
            elif use_global_threshold:
                event_properties['global_std'] = global_std
            else:
                event_properties['channel_std'] = threshold / threshold_multiplier
            
            events_list.append(event_properties)
    
    strategy_name = f"{threshold_strategy} (fixed={fixed_std_value:.4f})" if use_fixed_std else threshold_strategy
    print(f"Detected {len(events_list)} {data_type} events using {strategy_name}")
    
    # Return threshold info for plotting
    threshold_data = np.tile(np.array(channel_thresholds)[:, np.newaxis], (1, n_timepoints))
    
    return events_list, data_segment, threshold_data


def plot_events_with_arrows_global_fixed(raw_segment, processed_segment, events_list, 
                                        segment_name, toxin, trial_string, sampling_rate_hz=5, 
                                        save_dir_event_plots=None, max_channels_plot=10,
                                        threshold_multiplier=2.5, show_thresholds=True, use_global_threshold=False,
                                        use_fixed_std=True, fixed_std_value=None, data_type="voltage"):
    """
    FIXED: Plot timeseries with CORRECT baseline visualization (per-channel baselines + global threshold)
    """
    if raw_segment is None:
        print(f"Cannot plot {data_type} {segment_name} segment - data is None")
        return None
    
    n_channels, n_timepoints = raw_segment.shape
    time_axis = np.arange(n_timepoints) / sampling_rate_hz
    
    n_channels_plot = n_channels
    
    fig, ax = plt.subplots(figsize=(10, 20))
    
    # Calculate channel spacing
    data_ranges = [np.ptp(raw_segment[i, :]) for i in range(n_channels_plot)]
    max_range = max(data_ranges) if data_ranges else 1
    channel_spacing = max_range * 1.5
    
    # Calculate thresholds for visualization - MATCH the detection logic exactly
    if show_thresholds and processed_segment is not None:
        if use_fixed_std:
            print(f"FIXED: Using per-channel baselines + fixed STD threshold for {data_type} visualization...")
            
            # Use the provided fixed STD value or calculate from global data
            if fixed_std_value is None:
                all_data_flattened = processed_segment.flatten()
                fixed_std_value = np.std(all_data_flattened)
            
            fixed_threshold = threshold_multiplier * fixed_std_value
            
            print(f"  Fixed {data_type} std: {fixed_std_value:.4f}")
            print(f"  Fixed {data_type} threshold: ±{fixed_threshold:.4f}")
            print(f"  Using per-channel baselines (as in detection)")
            
        elif use_global_threshold:
            print(f"FIXED: Using per-channel baselines + global threshold for {data_type} visualization...")
            
            # Calculate global statistics (same as detection)
            all_data_flattened = processed_segment.flatten()
            global_mean = np.mean(all_data_flattened)
            global_std = np.std(all_data_flattened)
            global_threshold = threshold_multiplier * global_std
            
            print(f"  Global {data_type} std: {global_std:.4f}")
            print(f"  Global {data_type} threshold: ±{global_threshold:.4f}")
            print(f"  Using per-channel baselines (as in detection)")
            
        else:
            print(f"Using per-channel baselines + per-channel thresholds for {data_type} visualization...")
    
    for i in range(n_channels_plot):
        offset = i * channel_spacing
        
        # Plot raw data (grey) and processed data (black)
        ax.plot(time_axis, raw_segment[i, :] + offset, color='grey', 
               linewidth=0.8, alpha=0.7, label='Normalized + High-pass' if i == 0 else '')
        
        if processed_segment is not None:
            ax.plot(time_axis, processed_segment[i, :] + offset, color='black', 
                   linewidth=1.2, alpha=0.9, label='Final Processed' if i == 0 else '')
        
        # Add channel label
        ax.text(-0.02 * np.max(time_axis), np.mean(raw_segment[i, :]) + offset, f'{i}', 
                verticalalignment='center', fontsize=9, fontweight='bold')
        
        # Add threshold lines - FIXED TO MATCH DETECTION LOGIC
        if show_thresholds and processed_segment is not None:
            # ALWAYS use per-channel baseline (this matches the detection algorithm)
            channel_data = processed_segment[i, :]
            channel_baseline = np.median(channel_data)  # Per-channel baseline (median, same as detection)
            
            if use_fixed_std:
                # Use fixed STD threshold (same as detection)
                threshold_value = fixed_threshold
                threshold_label = f'Fixed STD Threshold (±{threshold_multiplier}σ)'
                baseline_label = 'Per-Channel Baseline (median)'
            elif use_global_threshold:
                # Use global threshold (same as detection)
                threshold_value = global_threshold
                threshold_label = f'Global Threshold (±{threshold_multiplier}σ)'
                baseline_label = 'Per-Channel Baseline (median)'
            else:
                # Use per-channel threshold
                channel_std = np.std(channel_data)
                threshold_value = threshold_multiplier * channel_std
                threshold_label = f'Per-Channel Threshold (±{threshold_multiplier}σ)'
                baseline_label = 'Per-Channel Baseline (median)'
            
            # Plot baseline (green) - ALWAYS per-channel
            ax.axhline(y=channel_baseline + offset, color='green', linestyle='-', 
                      linewidth=1.5, alpha=0.8, 
                      label=baseline_label if i == 0 else '')
            
            # Plot positive threshold (red)
            ax.axhline(y=channel_baseline + threshold_value + offset, 
                      color='red', linestyle='--', linewidth=1, alpha=0.8,
                      label=f'+{threshold_label}' if i == 0 else '')
            
            # Plot negative threshold (blue)
            ax.axhline(y=channel_baseline - threshold_value + offset, 
                      color='blue', linestyle='--', linewidth=1, alpha=0.8,
                      label=f'-{threshold_label}' if i == 0 else '')
        
        # Add event markers and spans for this channel
        channel_events = [event for event in events_list if event['channel_idx'] == i]
        
        for event in channel_events:
            event_start_time = event['start_time_sec']
            event_end_time = event['end_time_sec']
            
            # Get y-positions for event boundaries
            if processed_segment is not None:
                y_start = processed_segment[i, event['start_sample']] + offset
                y_end = processed_segment[i, event['end_sample']] + offset
            else:
                y_start = raw_segment[i, event['start_sample']] + offset
                y_end = raw_segment[i, event['end_sample']] + offset
            
            # Color based on event type
            event_color = 'red' if event['event_type'] == 'positive' else 'blue'
            
            # Add event span (horizontal line showing duration)
            span_y = offset + channel_spacing * 0.4
            ax.plot([event_start_time, event_end_time], [span_y, span_y], 
                   color=event_color, linewidth=3, alpha=0.7)
            
            # Add start and end markers
            ax.plot(event_start_time, y_start, 'o', color=event_color, markersize=4, alpha=0.9)
            ax.plot(event_end_time, y_end, 's', color=event_color, markersize=4, alpha=0.9)
    
    # Customize plot
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    
    # FIXED TITLE: Correctly describe what's being shown
    if use_fixed_std:
        title_text = f'{segment_name.upper()} Segment - {data_type.upper()} - {toxin} {trial_string}\n(Per-Channel Baselines (median) + Fixed STD Threshold Event Detection)'
    elif use_global_threshold:
        title_text = f'{segment_name.upper()} Segment - {data_type.upper()} - {toxin} {trial_string}\n(Per-Channel Baselines (median) + Global Threshold Event Detection)'
    else:
        title_text = f'{segment_name.upper()} Segment - {data_type.upper()} - {toxin} {trial_string}\n(Per-Channel Baselines (median) + Per-Channel Threshold Event Detection)'
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, np.max(time_axis))
    ax.set_ylim(-channel_spacing*0.5, (n_channels_plot-0.5) * channel_spacing)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    if n_channels_plot > 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add event count text with CORRECT threshold info
    total_events = len(events_list)
    positive_events = len([e for e in events_list if e['event_type'] == 'positive'])
    negative_events = len([e for e in events_list if e['event_type'] == 'negative'])
    
    if total_events > 0:
        avg_duration = np.mean([e['duration_sec'] for e in events_list])
        if use_fixed_std:
            event_text = f'Events: {total_events} total ({positive_events} pos, {negative_events} neg)\nPer-channel baselines (median) + Fixed STD threshold: ±{fixed_threshold:.4f}\nAvg duration: {avg_duration:.2f}s'
        elif use_global_threshold:
            event_text = f'Events: {total_events} total ({positive_events} pos, {negative_events} neg)\nPer-channel baselines (median) + Global threshold: ±{global_threshold:.4f}\nAvg duration: {avg_duration:.2f}s'
        else:
            event_text = f'Events: {total_events} total ({positive_events} pos, {negative_events} neg)\nPer-channel baselines (median) + Per-channel thresholds\nAvg duration: {avg_duration:.2f}s'
    else:
        if use_fixed_std:
            event_text = f'Events: 0 total\nPer-channel baselines (median) + Fixed STD threshold: ±{fixed_threshold:.4f}'
        elif use_global_threshold:
            event_text = f'Events: 0 total\nPer-channel baselines (median) + Global threshold: ±{global_threshold:.4f}'
        else:
            event_text = f'Events: 0 total\nPer-channel baselines (median) + Per-channel thresholds'
    
    ax.text(0.02, 0.98, event_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if save_dir_event_plots is not None:
        suffix = "global" if use_global_threshold else "perchannel"
        plot_path = Path(save_dir_event_plots) / f"{segment_name}_{data_type}_{toxin}_{trial_string}_events_{suffix}_fixed.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  FIXED {data_type} plot saved to: {plot_path}")
    
    plt.close()
    return plot_path

def filter_simultaneous_events(events_list, max_simultaneous_cells=3, overlap_threshold=0.5, data_type="voltage"):
    """
    Filter out simultaneous events that might be artifacts (dust, focal shifts, etc.)
    """
    if not events_list:
        return [], []
    
    print(f"Filtering simultaneous {data_type} events (max {max_simultaneous_cells} cells, {overlap_threshold*100:.0f}% overlap)...")
    
    # Create a list to track which events to exclude
    events_to_exclude = set()
    
    # Check each event against all other events
    for i, event1 in enumerate(events_list):
        if i in events_to_exclude:
            continue
            
        # Find all events that overlap with this event
        overlapping_events = []
        
        for j, event2 in enumerate(events_list):
            if i == j or j in events_to_exclude:
                continue
                
            # Calculate overlap between event1 and event2
            overlap_fraction = calculate_overlap_fraction(event1, event2)
            
            if overlap_fraction >= overlap_threshold:
                overlapping_events.append(j)
        
        # If too many cells have overlapping events, mark them all for exclusion
        if len(overlapping_events) + 1 > max_simultaneous_cells:  # +1 includes the current event
            print(f"  Found {len(overlapping_events) + 1} simultaneous {data_type} events at {event1['start_time_sec']:.1f}s - marking for exclusion")
            events_to_exclude.add(i)
            events_to_exclude.update(overlapping_events)
    
    # Split events into kept and excluded
    filtered_events = []
    excluded_events = []
    
    for i, event in enumerate(events_list):
        if i in events_to_exclude:
            # Add exclusion reason to event
            event_copy = event.copy()
            event_copy['exclusion_reason'] = 'simultaneous_artifact'
            excluded_events.append(event_copy)
        else:
            filtered_events.append(event)
    
    print(f"  Original {data_type} events: {len(events_list)}")
    print(f"  Filtered {data_type} events: {len(filtered_events)}")
    print(f"  Excluded {data_type} events: {len(excluded_events)}")
    
    return filtered_events, excluded_events

def calculate_overlap_fraction(event1, event2):
    """
    Calculate the fraction of overlap between two events
    """
    start1, end1 = event1['start_time_sec'], event1['end_time_sec']
    start2, end2 = event2['start_time_sec'], event2['end_time_sec']
    
    # Calculate overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0  # No overlap
    
    overlap_duration = overlap_end - overlap_start
    
    # Calculate overlap as fraction of the shorter event
    duration1 = end1 - start1
    duration2 = end2 - start2
    shorter_duration = min(duration1, duration2)
    
    if shorter_duration <= 0:
        return 0.0
    
    overlap_fraction = overlap_duration / shorter_duration
    return overlap_fraction

def process_events_for_segments_with_enhanced_filter(segments_raw, segments_processed, toxin, trial_string, 
                                                   save_dir_events, save_dir_event_plots, sampling_rate_hz=5,
                                                   max_simultaneous_percentage=0.5, overlap_threshold=0.5, 
                                                   has_post_in_name=False, data_type="voltage", 
                                                   cell_positions=None, **event_detection_options):
    """
    Process segments for event detection with bleaching artifact and simultaneous event filtering
    """
    all_events = []
    all_excluded_events = []
    
    # Extract filtering parameters
    max_simultaneous_cells = event_detection_options.get('max_simultaneous_cells', 10)
    filter_bleaching = event_detection_options.get('filter_bleaching_artifacts', True)
    bleaching_min_cells = event_detection_options.get('bleaching_min_cells', 5)
    bleaching_overlap_threshold = event_detection_options.get('bleaching_overlap_threshold', 0.7)
    bleaching_start_time_threshold = event_detection_options.get('bleaching_start_time_threshold', 10.0)
    
    # ADD THESE MISSING VARIABLES:
    use_fixed = event_detection_options.get('use_fixed_std', True)
    use_global = event_detection_options.get('use_global_threshold', False)
    
    for segment_name in segments_processed.keys():
        raw_segment = segments_raw.get(segment_name)
        processed_segment = segments_processed.get(segment_name)
        
        if processed_segment is None:
            continue
        
        # Determine correct segment label
        if has_post_in_name:
            correct_segment = 'post'
        else:
            correct_segment = segment_name
            
        # Determine threshold strategy (now use_fixed and use_global are defined)
        if use_fixed:
            threshold_type = "FIXED STD"
        elif use_global:
            threshold_type = "GLOBAL STD"
        else:
            threshold_type = "PER-CHANNEL STD"
            
        events_list, filtered_data, threshold_data = detect_voltage_events_enhanced_threshold(
            processed_segment, 
            sampling_rate_hz=sampling_rate_hz,
            cell_id_mapping=cell_positions,  # Use the cell_positions parameter
            data_type=data_type,
            **event_detection_options
        )
        
        # Add metadata to events
        for event in events_list:
            event.update({
                'trial_string': trial_string,
                'toxin': toxin,
                'segment': correct_segment,
                'cell_index': event['channel_idx'],
                'data_type': data_type
            })
        
        # Step 1: Filter bleaching artifacts (if enabled)
        if filter_bleaching:
            print(f"\n=== FILTERING BLEACHING ARTIFACTS: {data_type.upper()} {segment_name.upper()} SEGMENT ===")
            events_after_bleaching, bleaching_excluded = filter_bleaching_artifacts(
                events_list,
                min_cells_threshold=bleaching_min_cells,
                overlap_threshold=bleaching_overlap_threshold,
                start_time_threshold=bleaching_start_time_threshold,
                data_type=data_type
            )
            all_excluded_events.extend(bleaching_excluded)
        else:
            events_after_bleaching = events_list
        
        # Step 2: Filter simultaneous events
        print(f"\n=== FILTERING SIMULTANEOUS EVENTS: {data_type.upper()} {segment_name.upper()} SEGMENT ===")
        filtered_events, simultaneous_excluded = filter_simultaneous_events(
            events_after_bleaching, 
            max_simultaneous_cells=max_simultaneous_cells,
            data_type=data_type
        )
        
        all_events.extend(filtered_events)
        all_excluded_events.extend(simultaneous_excluded)
        
        # Create plot with filtered events
        plot_events_with_arrows_global_fixed(
            raw_segment, processed_segment, filtered_events, segment_name,
            toxin, trial_string, sampling_rate_hz, save_dir_event_plots,
            threshold_multiplier=event_detection_options.get('threshold_multiplier', 2.5),
            show_thresholds=True,
            use_global_threshold=event_detection_options.get('use_global_threshold', False),
            use_fixed_std=event_detection_options.get('use_fixed_std', True),
            fixed_std_value=event_detection_options.get('fixed_std_value'),
            data_type=data_type
        )
    
    # Save filtered events to CSV with enhanced naming
    if all_events:
        events_df = pd.DataFrame(all_events)
        
        # Enhanced column order to include new fields
        column_order = ['trial_string', 'toxin', 'segment', 'cell_index', 'original_cell_id', 'data_type',
                    'start_time_sec', 'end_time_sec', 'duration_sec', 
                    'amplitude', 'event_type', 'mean_amplitude', 'peak_value',
                    'baseline_value', 'threshold_used', 'threshold_type',
                    'start_sample', 'end_sample', 'duration_samples',
                    'threshold_start_sample', 'threshold_end_sample']
        
        # Add strategy-specific columns if they exist
        if 'fixed_std_value' in events_df.columns:
            column_order.append('fixed_std_value')
        if 'global_std' in events_df.columns:
            column_order.append('global_std')
        if 'channel_std' in events_df.columns:
            column_order.append('channel_std')
        
        events_df = events_df[column_order]
        
        # Save with appropriate suffix
        if use_fixed:
            suffix = "fixed_std"
        elif use_global:
            suffix = "global"
        else:
            suffix = "perchannel"
            
        events_path = Path(save_dir_events) / f"events_{data_type}_{toxin}_{trial_string}_{suffix}_adaptive_filtered.csv"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(events_path, index=False)
        print(f"Enhanced filtered {data_type} events saved to: {events_path}")
        
        # Print summary
        durations = events_df['duration_sec'].values
        print(f"\n{threshold_type} {data_type.upper()} Event Summary for {toxin} {trial_string}:")
        print(f"  Total filtered events: {len(all_events)}")
        print(f"  Total excluded events: {len(all_excluded_events)} ({max_simultaneous_percentage*100:.0f}% rule)")
        print(f"  Pre-segment events: {len([e for e in all_events if e['segment'] == 'pre'])}")
        print(f"  Post-segment events: {len([e for e in all_events if e['segment'] == 'post'])}")
        if len(durations) > 0:
            print(f"  Duration stats: mean={np.mean(durations):.2f}s, median={np.median(durations):.2f}s")
        
        return events_df, all_excluded_events
    
    return None, all_excluded_events

def filter_bleaching_artifacts(events_list, min_cells_threshold=5, overlap_threshold=0.7, 
                              start_time_threshold=10.0, data_type="voltage"):
    """
    Filter out events that appear to be bleaching artifacts at recording start
    
    Criteria:
    - More than min_cells_threshold cells have same polarity events
    - Events overlap by more than overlap_threshold (70%)
    - Events start within start_time_threshold seconds of recording start
    """
    if not events_list:
        return [], []
    
    print(f"Filtering bleaching artifacts for {data_type} (≥{min_cells_threshold} cells, {overlap_threshold*100:.0f}% overlap, start <{start_time_threshold}s)...")
    
    # Group events by polarity (positive/negative)
    positive_events = [e for e in events_list if e['event_type'] == 'positive']
    negative_events = [e for e in events_list if e['event_type'] == 'negative']
    
    bleaching_event_indices = set()
    
    for event_group, polarity in [(positive_events, 'positive'), (negative_events, 'negative')]:
        if len(event_group) < min_cells_threshold:
            continue
            
        # Filter events that start near beginning of recording
        early_events = [e for e in event_group if e['start_time_sec'] <= start_time_threshold]
        
        if len(early_events) < min_cells_threshold:
            continue
        
        # Check for high overlap among early events
        overlapping_groups = find_overlapping_event_groups(early_events, overlap_threshold)
        
        for group in overlapping_groups:
            if len(group) >= min_cells_threshold:
                # Mark these events as bleaching artifacts
                group_indices = [events_list.index(event) for event in group if event in events_list]
                bleaching_event_indices.update(group_indices)
                
                start_times = [e['start_time_sec'] for e in group]
                cell_indices = [e['cell_index'] for e in group]
                print(f"  Found {len(group)} {polarity} bleaching events (cells: {cell_indices}, times: {min(start_times):.1f}-{max(start_times):.1f}s)")
    
    # Split events into kept and excluded
    filtered_events = []
    excluded_events = []
    
    for i, event in enumerate(events_list):
        if i in bleaching_event_indices:
            event_copy = event.copy()
            event_copy['exclusion_reason'] = 'bleaching_artifact'
            excluded_events.append(event_copy)
        else:
            filtered_events.append(event)
    
    print(f"  Original {data_type} events: {len(events_list)}")
    print(f"  Excluded bleaching artifacts: {len(excluded_events)}")
    print(f"  Remaining {data_type} events: {len(filtered_events)}")
    
    return filtered_events, excluded_events

def find_overlapping_event_groups(events, overlap_threshold):
    """
    Find groups of events that have significant temporal overlap
    """
    if len(events) < 2:
        return []
    
    # Create overlap matrix
    n_events = len(events)
    overlap_matrix = np.zeros((n_events, n_events))
    
    for i in range(n_events):
        for j in range(i+1, n_events):
            overlap_frac = calculate_overlap_fraction(events[i], events[j])
            overlap_matrix[i, j] = overlap_frac
            overlap_matrix[j, i] = overlap_frac
    
    # Find connected components (groups of overlapping events)
    visited = [False] * n_events
    overlapping_groups = []
    
    for i in range(n_events):
        if visited[i]:
            continue
            
        # Start a new group
        current_group = []
        stack = [i]
        
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
                
            visited[current] = True
            current_group.append(events[current])
            
            # Add all events that overlap significantly with current event
            for j in range(n_events):
                if not visited[j] and overlap_matrix[current, j] >= overlap_threshold:
                    stack.append(j)
        
        if len(current_group) > 1:
            overlapping_groups.append(current_group)
    
    return overlapping_groups

def save_excluded_events(excluded_events, toxin, trial_string, save_dir_events, data_type="voltage"):
    """
    Save excluded events to a separate CSV for review
    """
    if excluded_events:
        excluded_df = pd.DataFrame(excluded_events)
        
        # Add the exclusion reason column
        column_order = ['trial_string', 'toxin', 'segment', 'cell_index', 'data_type',
                       'start_time_sec', 'end_time_sec', 'duration_sec', 
                       'amplitude', 'event_type', 'exclusion_reason',
                       'baseline_value', 'threshold_used', 'threshold_type',
                       'start_sample', 'end_sample', 'duration_samples']
        
        excluded_df = excluded_df[column_order]
        
        # Save excluded events
        excluded_path = Path(save_dir_events) / f"excluded_events_{data_type}_{toxin}_{trial_string}.csv"
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_df.to_csv(excluded_path, index=False)
        print(f"Excluded {data_type} events saved to: {excluded_path}")
        
        return excluded_path
    return None

def process_single_trial_complete(voltage_file, ca_file, toxin, trial_string, 
                                save_dir_plots, save_dir_data, save_dir_events, save_dir_event_plots,
                                apply_mean_center=True, apply_detrend=True, 
                                apply_gaussian=False, gaussian_sigma=3,
                                apply_normalization=True,
                                enable_event_detection=True, **event_options):
    """
    Complete flexible workflow for BOTH calcium and voltage data with file type detection
    """
    print(f"\nProcessing trial: {trial_string}")
    print(f"Voltage file: {voltage_file.name}")
    print(f"Calcium file: {ca_file.name}")
    print("-" * 80)
    
    try:
        # Load both datasets
        print("Loading voltage and calcium data...")
        voltage_data = pd.read_csv(voltage_file)
        ca_data = pd.read_csv(ca_file)
        
        # Detect file types
        voltage_file_type, voltage_n_datapoints = detect_file_type_and_datapoints(voltage_data)
        ca_file_type, ca_n_datapoints = detect_file_type_and_datapoints(ca_data)
        
        # Extract cell positions AND original cell IDs BEFORE processing
        voltage_cell_positions = voltage_data[['cell_id', 'cell_x', 'cell_y']].copy()  # Include cell_Id
        ca_cell_positions = ca_data[['cell_id', 'cell_x', 'cell_y']].copy()

        # Extract timeseries data (everything after cell_y)
        cell_y_idx_v = voltage_data.columns.get_loc('cell_y')
        cell_y_idx_c = ca_data.columns.get_loc('cell_y')
        voltage_timeseries = voltage_data.iloc[:, cell_y_idx_v + 1:]
        ca_timeseries = ca_data.iloc[:, cell_y_idx_c + 1:]

        print(f"Original voltage cell IDs: {voltage_cell_positions['cell_id'].tolist()}")
        print(f"Original calcium cell IDs: {ca_cell_positions['cell_id'].tolist()}")
        
        # Check for 'post' in filename
        has_post_in_name = 'post' in voltage_file.name.lower() or 'post' in ca_file.name.lower()
        
        if voltage_file_type == "post_only" or has_post_in_name:
            print("=== POST-ONLY FILES DETECTED - SKIPPING SEGMENTATION ===")
            
            # Apply normalization + high-pass filter to BOTH datasets
            voltage_highpass = apply_highpass_to_full_timeseries(
                voltage_timeseries, 
                sampling_rate_hz=5, 
                cutoff_freq=0.01,
                apply_normalization=apply_normalization,
                data_type="voltage"
            )
            
            ca_highpass = apply_highpass_to_full_timeseries(
                ca_timeseries, 
                sampling_rate_hz=5, 
                cutoff_freq=0.01,
                apply_normalization=apply_normalization,
                data_type="calcium"
            )
            
            # Create segments dictionary - only 'post' segment
            voltage_segments_raw = {'post': voltage_highpass}
            ca_segments_raw = {'post': ca_highpass}
            voltage_segments_processed = {}
            ca_segments_processed = {}
            
            # Process the single 'post' segment
            for segment_name in ['post']:
                # Process VOLTAGE
                print(f"\n=== PROCESSING VOLTAGE {segment_name.upper()} SEGMENT ===")
                voltage_processed, voltage_non_gaussian = apply_segment_processing(
                    voltage_segments_raw[segment_name],
                    apply_mean_center=apply_mean_center,
                    apply_detrend=apply_detrend,
                    apply_gaussian=apply_gaussian,
                    gaussian_sigma=gaussian_sigma,
                    data_type="voltage"
                )
                
                voltage_segments_processed[segment_name] = voltage_processed
                
                # Save with 'post' label (no pre/post confusion)
                save_segment_data(voltage_processed, 'post', toxin, trial_string, 
                                 save_dir_data, cell_positions=voltage_cell_positions, 
                                 original_filename=voltage_file.name, data_type="voltage")
                
                # Create comparison plot
                plot_raw_and_filtered_segment(
                    voltage_segments_raw[segment_name], voltage_processed, voltage_non_gaussian,
                    'post', toxin, trial_string, 
                    save_dir_plots=save_dir_plots, data_type="voltage"
                )
                
                # Process CALCIUM
                print(f"\n=== PROCESSING CALCIUM {segment_name.upper()} SEGMENT ===")
                ca_processed, ca_non_gaussian = apply_segment_processing(
                    ca_segments_raw[segment_name],
                    apply_mean_center=apply_mean_center,
                    apply_detrend=apply_detrend,
                    apply_gaussian=apply_gaussian,
                    gaussian_sigma=gaussian_sigma,
                    data_type="calcium"
                )
                
                ca_segments_processed[segment_name] = ca_processed
                
                save_segment_data(ca_processed, 'post', toxin, trial_string, 
                                 save_dir_data, cell_positions=ca_cell_positions, 
                                 original_filename=ca_file.name, data_type="calcium")
                
                plot_raw_and_filtered_segment(
                    ca_segments_raw[segment_name], ca_processed, ca_non_gaussian,
                    'post', toxin, trial_string, 
                    save_dir_plots=save_dir_plots, data_type="calcium"
                )
            
        else:
            print("=== FULL EXPERIMENT FILES DETECTED - USING SEGMENTATION ===")
            
            # Original pipeline: normalization → consensus → segmentation → processing
            voltage_highpass = apply_highpass_to_full_timeseries(
                voltage_timeseries, 
                sampling_rate_hz=5, 
                cutoff_freq=0.01,
                apply_normalization=apply_normalization,
                data_type="voltage"
            )
            
            ca_highpass = apply_highpass_to_full_timeseries(
                ca_timeseries, 
                sampling_rate_hz=5, 
                cutoff_freq=0.01,
                apply_normalization=apply_normalization,
                data_type="calcium"
            )
            
            # Find consensus timepoint using VOLTAGE data
            consensus_timepoint = find_consensus_timepoint_from_voltage(
                voltage_highpass,
                sampling_rate_hz=5,
                time_window=(4500, 5500),
                detection_method='zscore',
                consensus_method='mode'
            )
            
            if consensus_timepoint is None:
                print("✗ WARNING: No consensus found - cannot segment")
                return False
            
            if consensus_timepoint is not None and trial_metadata:
                create_segment_videos(trial_string, consensus_timepoint, trial_metadata, 
                                    save_dir_data, sampling_rate_hz=5)
            
            # Segment both datasets using the same consensus timepoint
            voltage_segments_raw = segment_highpass_timeseries(voltage_highpass, consensus_timepoint, data_type="voltage")
            ca_segments_raw = segment_highpass_timeseries(ca_highpass, consensus_timepoint, data_type="calcium")
            voltage_segments_processed = {}
            ca_segments_processed = {}
            
            # Process each segment (pre and post)
            for segment_name in ['pre', 'post']:
                # Process voltage segments
                if voltage_segments_raw[segment_name] is not None:
                    print(f"\n=== PROCESSING VOLTAGE {segment_name.upper()} SEGMENT ===")
                    
                    voltage_processed, voltage_non_gaussian = apply_segment_processing(
                        voltage_segments_raw[segment_name],
                        apply_mean_center=apply_mean_center,
                        apply_detrend=apply_detrend,
                        apply_gaussian=apply_gaussian,
                        gaussian_sigma=gaussian_sigma,
                        data_type="voltage"
                    )
                    
                    voltage_segments_processed[segment_name] = voltage_processed
                    
                    # Save with original segment name (pre/post)
                    save_segment_data(voltage_processed, segment_name, toxin, trial_string, 
                                     save_dir_data, cell_positions=voltage_cell_positions, 
                                     original_filename=voltage_file.name, data_type="voltage")
                    
                    plot_raw_and_filtered_segment(
                        voltage_segments_raw[segment_name], voltage_processed, voltage_non_gaussian,
                        segment_name, toxin, trial_string, 
                        save_dir_plots=save_dir_plots, data_type="voltage"
                    )
                
                # Process calcium segments
                if ca_segments_raw[segment_name] is not None:
                    print(f"\n=== PROCESSING CALCIUM {segment_name.upper()} SEGMENT ===")
                    
                    ca_processed, ca_non_gaussian = apply_segment_processing(
                        ca_segments_raw[segment_name],
                        apply_mean_center=apply_mean_center,
                        apply_detrend=apply_detrend,
                        apply_gaussian=apply_gaussian,
                        gaussian_sigma=gaussian_sigma,
                        data_type="calcium"
                    )
                    
                    ca_segments_processed[segment_name] = ca_processed
                    
                    save_segment_data(ca_processed, segment_name, toxin, trial_string, 
                                     save_dir_data, cell_positions=ca_cell_positions, 
                                     original_filename=ca_file.name, data_type="calcium")
                    
                    plot_raw_and_filtered_segment(
                        ca_segments_raw[segment_name], ca_processed, ca_non_gaussian,
                        segment_name, toxin, trial_string, 
                        save_dir_plots=save_dir_plots, data_type="calcium"
                    )
        
        # Event detection (same for both cases)
        if enable_event_detection:
            print(f"\n=== VOLTAGE EVENT DETECTION ===")
            voltage_events_df, voltage_excluded = process_events_for_segments_with_enhanced_filter(
                voltage_segments_raw, voltage_segments_processed, toxin, trial_string,
                save_dir_events, save_dir_event_plots,
                sampling_rate_hz=5, data_type="voltage", 
                cell_positions=voltage_cell_positions,  # ADD THIS LINE
                has_post_in_name=has_post_in_name, **event_options
            )

            print(f"\n=== CALCIUM EVENT DETECTION ===")
            ca_events_df, ca_excluded = process_events_for_segments_with_enhanced_filter(
                ca_segments_raw, ca_segments_processed, toxin, trial_string,
                save_dir_events, save_dir_event_plots,
                sampling_rate_hz=5, data_type="calcium", 
                cell_positions=ca_cell_positions,  # ADD THIS LINE
                has_post_in_name=has_post_in_name, **event_options
            )
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR processing {trial_string}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def combine_all_events(base_results_dir, data_type="voltage", output_filename=None):
    """
    Combine all individual event CSV files into one master file for each data type
    Now searches in trial-specific directories
    """
    if output_filename is None:
        output_filename = f"all_events_combined_{data_type}_global.csv"
    
    # Search for event files in all trial subdirectories
    event_files = []
    for trial_dir in base_results_dir.iterdir():
        if trial_dir.is_dir():
            trial_event_files = list(trial_dir.glob(f"events_{data_type}_*_global_filtered.csv"))
            event_files.extend(trial_event_files)
    
    if not event_files:
        print(f"No {data_type} global threshold event files found to combine")
        return None
    
    all_dataframes = []
    for file in event_files:
        df = pd.read_csv(file)
        all_dataframes.append(df)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save combined file in base results directory
    combined_path = base_results_dir / output_filename
    combined_df.to_csv(combined_path, index=False)
    
    print(f"\nCombined {data_type} events file saved: {combined_path}")
    print(f"Total {data_type} events across all experiments: {len(combined_df)}")
    
    # Print summary statistics
    print(f"\n{data_type.capitalize()} summary by toxin:")
    print(combined_df.groupby(['toxin', 'segment']).size().unstack(fill_value=0))
    
    return combined_df

def create_summary_plots(base_results_dir, save_dir_plots):
    """
    Create summary comparison plots between calcium and voltage events
    """
    print("\n=== CREATING SUMMARY COMPARISON PLOTS ===")
    
    # Try to load combined event files from base results directory
    voltage_file = Path(base_results_dir) / "all_events_combined_voltage_global.csv"
    calcium_file = Path(base_results_dir) / "all_events_combined_calcium_global.csv"
    
    if not voltage_file.exists() or not calcium_file.exists():
        print("Combined event files not found - skipping summary plots")
        return
    
    voltage_df = pd.read_csv(voltage_file)
    calcium_df = pd.read_csv(calcium_file)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Event counts by toxin and segment
    ax1 = axes[0, 0]
    voltage_counts = voltage_df.groupby(['toxin', 'segment']).size().unstack(fill_value=0)
    calcium_counts = calcium_df.groupby(['toxin', 'segment']).size().unstack(fill_value=0)
    
    x = np.arange(len(voltage_counts.index))
    width = 0.35
    
    ax1.bar(x - width/2, voltage_counts['pre'], width, label='Voltage Pre', alpha=0.8, color='blue')
    ax1.bar(x + width/2, calcium_counts['pre'], width, label='Calcium Pre', alpha=0.8, color='red')
    ax1.set_xlabel('Toxin')
    ax1.set_ylabel('Event Count (Pre)')
    ax1.set_title('Pre-segment Event Counts')
    ax1.set_xticks(x)
    ax1.set_xticklabels(voltage_counts.index, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Post-segment events
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, voltage_counts['post'], width, label='Voltage Post', alpha=0.8, color='darkblue')
    ax2.bar(x + width/2, calcium_counts['post'], width, label='Calcium Post', alpha=0.8, color='darkred')
    ax2.set_xlabel('Toxin')
    ax2.set_ylabel('Event Count (Post)')
    ax2.set_title('Post-segment Event Counts')
    ax2.set_xticks(x)
    ax2.set_xticklabels(voltage_counts.index, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Duration comparison
    ax3 = axes[1, 0]
    voltage_durations = voltage_df['duration_sec']
    calcium_durations = calcium_df['duration_sec']
    
    ax3.hist(voltage_durations, bins=30, alpha=0.7, label=f'Voltage (n={len(voltage_durations)})', color='blue')
    ax3.hist(calcium_durations, bins=30, alpha=0.7, label=f'Calcium (n={len(calcium_durations)})', color='red')
    ax3.set_xlabel('Event Duration (s)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Event Duration Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Amplitude comparison
    ax4 = axes[1, 1]
    voltage_amplitudes = voltage_df['amplitude']
    calcium_amplitudes = calcium_df['amplitude']
    
    ax4.hist(voltage_amplitudes, bins=30, alpha=0.7, label=f'Voltage (n={len(voltage_amplitudes)})', color='blue')
    ax4.hist(calcium_amplitudes, bins=30, alpha=0.7, label=f'Calcium (n={len(calcium_amplitudes)})', color='red')
    ax4.set_xlabel('Event Amplitude')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Event Amplitude Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = Path(save_dir_plots) / "calcium_voltage_event_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Summary comparison plot saved to: {summary_path}")
    plt.close()
    
    # Print summary statistics
    print("\n=== EVENT DETECTION SUMMARY ===")
    print(f"Total voltage events: {len(voltage_df)}")
    print(f"Total calcium events: {len(calcium_df)}")
    print(f"Voltage event rate: {len(voltage_df)/len(voltage_df['trial_string'].unique()):.1f} events/trial")
    print(f"Calcium event rate: {len(calcium_df)/len(calcium_df['trial_string'].unique()):.1f} events/trial")
    
    if len(voltage_durations) > 0 and len(calcium_durations) > 0:
        print(f"Voltage duration: {np.mean(voltage_durations):.2f}±{np.std(voltage_durations):.2f}s")
        print(f"Calcium duration: {np.mean(calcium_durations):.2f}±{np.std(calcium_durations):.2f}s")

def main():
    """
    Main workflow with integrated calcium and voltage analysis
    """
    # Set up directories
    home = Path.home()
    if "ys5320" in str(home):
        top_dir = Path(home, "firefly_link")
    else:
        top_dir = Path(r'R:\home\firefly_link')
    
    # NEW DIRECTORY STRUCTURE
    base_results_dir = Path(top_dir, r'Calcium_Voltage_Imaging\code_yilin\results_new')
    data_dir = Path(top_dir, r'ca_voltage_imaging_working\results')
    
    # Global directories for plots
    save_dir_plots = Path(base_results_dir, 'timeseries_plots')
    save_dir_event_plots = Path(base_results_dir, 'event_detection_plots')
    
    # Create global plot directories
    save_dir_plots.mkdir(parents=True, exist_ok=True)
    save_dir_event_plots.mkdir(parents=True, exist_ok=True)
    
    # Note: Trial-specific data directories will be created per trial
    
    # Processing options - normalization moved to very beginning
    PROCESSING_OPTIONS = {
        'apply_mean_center': False,
        'apply_detrend': False,
        'apply_gaussian': True,
        'gaussian_sigma': 3,
        'apply_normalization': True     # Applies at the very beginning!
    }
    
    EVENT_DETECTION_OPTIONS = {
        'enable_event_detection': True,
        'threshold_multiplier': 2.5,
        'apply_gaussian_for_detection': False,
        'min_event_duration_sec': 2,
        'use_global_threshold': False,
        'max_simultaneous_cells': 999,
        'overlap_threshold': 0.9,
        'use_fixed_std': True,
        'fixed_std_value': 0.05,
        
        # New bleaching artifact filtering
        'filter_bleaching_artifacts': True,
        'bleaching_min_cells': 5,           # Minimum cells for bleaching detection
        'bleaching_overlap_threshold': 0.7,  # 70% overlap threshold
        'bleaching_start_time_threshold': 10.0,  # Events starting within first 10 seconds
    }
    
    print("=== INTEGRATED CALCIUM-VOLTAGE WORKFLOW ===")
    print("1. Process voltage data → Find consensus timepoint")
    print("2. Use SAME consensus for calcium segmentation")
    print("3. Apply identical processing to both modalities")
    print("4. Global threshold event detection on both")
    print("🎯 SYNCHRONIZED: Same timing, same processing, same detection criteria")
    print()
    
    toxins = [
        #'L-15_control',
        'Ani9_10uM','ATP_1mM','TRAM-34_1uM','dantrolene_10uM']
    #toxins = ['L-15_control']
    total_successful = 0
    total_processed = 0
    
    print("Processing Options:")
    for key, value in PROCESSING_OPTIONS.items():
        print(f"  {key}: {value}")
    print()
    
    print("Event Detection Options:")
    for key, value in EVENT_DETECTION_OPTIONS.items():
        print(f"  {key}: {value}")
    print()
    
    # Find matching files for all toxins
    print("=== FINDING MATCHING CALCIUM-VOLTAGE FILE PAIRS ===")
    matched_files = find_matching_files(data_dir, toxins)
    
    # Validate file pairs
    valid_pairs, total_pairs = validate_file_pairs(matched_files)
    if valid_pairs == 0:
        print("❌ No valid file pairs found! Exiting...")
        return
    
    for toxin in toxins:
        print(f"\n{'='*80}")
        print(f"PROCESSING TOXIN: {toxin}")
        print(f"{'='*80}")
        
        trial_pairs = matched_files.get(toxin, [])
        
        if not trial_pairs:
            print(f"No matching pairs found for {toxin}!")
            continue
        
        print(f"Found {len(trial_pairs)} matching pairs for {toxin}")
        
        # Process each trial pair
        for trial_info in trial_pairs:
            total_processed += 1
            
            # Create trial-specific directory for data
            trial_data_dir = base_results_dir / trial_info['trial_string']
            trial_data_dir.mkdir(parents=True, exist_ok=True)
            
            success = process_single_trial_complete(
                trial_info['voltage_file'], 
                trial_info['ca_file'], 
                toxin, 
                trial_info['trial_string'],
                save_dir_plots, trial_data_dir, trial_data_dir, save_dir_event_plots,  # Updated paths
                **PROCESSING_OPTIONS, **EVENT_DETECTION_OPTIONS
            )
            if success:
                total_successful += 1
            print("-" * 80)
    
    # After processing all files, combine events if event detection was enabled
    if EVENT_DETECTION_OPTIONS['enable_event_detection']:
        print(f"\n{'='*80}")
        print("COMBINING ALL EVENTS")
        print(f"{'='*80}")
        
        # Combine voltage events
        voltage_combined = combine_all_events(base_results_dir, data_type="voltage")
        
        # Combine calcium events
        ca_combined = combine_all_events(base_results_dir, data_type="calcium")
        
        # Create summary comparison plots
        create_summary_plots(base_results_dir, save_dir_plots)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"INTEGRATED CA-VOLTAGE PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total trial pairs processed: {total_processed}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_processed - total_successful}")
    print(f"Data saved to trial-specific folders in: {base_results_dir}")
    print(f"Processing plots saved to: {save_dir_plots}")
    if EVENT_DETECTION_OPTIONS['enable_event_detection']:
        print(f"Event data saved in trial folders within: {base_results_dir}")
        print(f"Event plots saved to: {save_dir_event_plots}")
    
    print("\n=== INTEGRATED ANALYSIS SUMMARY ===")
    print("✅ Voltage-based consensus: Timing determined from voltage data")
    print("✅ Synchronized segmentation: Same timepoints for both modalities") 
    print("✅ Parallel processing: Identical pipeline for voltage & calcium")
    print("✅ Dual event detection: Global thresholds applied to both")
    print("✅ Comprehensive output: Side-by-side comparison possible")

if __name__ == "__main__":
    main()