import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
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

def apply_highpass_to_full_timeseries(data_matrix, sampling_rate_hz=5, cutoff_freq=0.01, apply_normalization=True):
    """
    Step 1: Apply normalization first, then high-pass filter to the entire timeseries
    """
    print("=== STEP 1: Normalization + High-Pass Filter to Full Timeseries ===")
    
    # Ensure data is numeric
    data_array = ensure_numeric_data(data_matrix)
    
    # Apply normalization FIRST (before any filtering)
    if apply_normalization:
        print("Applying normalization to raw data (before filtering)...")
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
        print(f"  - Min-max normalization applied (range: [-1, 1])")
    
    # Apply high-pass filter to normalized data
    print(f"Applying high-pass filter to normalized data (cutoff: {cutoff_freq} Hz)...")
    filtered_data = np.zeros_like(data_array)
    for i in range(data_array.shape[0]):
        filtered_data[i, :] = apply_highpass_filter(data_array[i, :], sampling_rate_hz, cutoff_freq)
    
    print(f"Normalization + High-pass filtering complete. Shape: {filtered_data.shape}")
    return filtered_data

def find_consensus_timepoint_only(highpass_data, sampling_rate_hz=5, time_window=(4500, 5500), 
                                 detection_method='zscore', consensus_method='mode'):
    """
    Step 2: Find consensus timepoint using high-pass filtered data
    """
    print("=== STEP 2: Finding Consensus Timepoint ===")
    
    # Apply mean centering for detection only (don't modify original data)
    detection_data = np.zeros_like(highpass_data)
    for i in range(highpass_data.shape[0]):
        channel_mean = np.mean(highpass_data[i, :])
        detection_data[i, :] = highpass_data[i, :] - channel_mean
    
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

def segment_highpass_timeseries(highpass_data, consensus_timepoint, sampling_rate_hz=5):
    """
    Step 3: Segment the high-pass filtered timeseries
    """
    print("=== STEP 3: Segmenting High-Pass Filtered Timeseries ===")
    
    n_channels, n_timepoints = highpass_data.shape
    
    # Define segment boundaries
    pre_start = 200
    pre_end = consensus_timepoint - 200
    post_start = consensus_timepoint + 400
    post_end = n_timepoints - 1
    
    # Validate and create segments
    segments = {}
    
    if pre_start >= pre_end:
        print(f"Warning: Pre-segment invalid (start: {pre_start}, end: {pre_end})")
        segments['pre'] = None
    else:
        segments['pre'] = highpass_data[:, pre_start:pre_end+1]
    
    if post_start >= post_end:
        print(f"Warning: Post-segment invalid (start: {post_start}, end: {post_end})")
        segments['post'] = None
    else:
        segments['post'] = highpass_data[:, post_start:post_end+1]
    
    # Print segment info
    samples_per_min = sampling_rate_hz * 60
    print(f"Consensus timepoint: {consensus_timepoint} frames ({consensus_timepoint/samples_per_min:.2f} min)")
    
    if segments['pre'] is not None:
        print(f"PRE segment: frames {pre_start}-{pre_end} ({pre_start/samples_per_min:.2f}-{pre_end/samples_per_min:.2f} min)")
        print(f"  Shape: {segments['pre'].shape}")
    
    if segments['post'] is not None:
        print(f"POST segment: frames {post_start}-{post_end} ({post_start/samples_per_min:.2f}-{post_end/samples_per_min:.2f} min)")
        print(f"  Shape: {segments['post'].shape}")
    
    return segments

def apply_segment_processing(segment_data, apply_mean_center=True, apply_detrend=True, 
                           apply_gaussian=False, gaussian_sigma=3):
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
        print("  - Mean centering applied")
    
    # Step 4b: Apply linear detrending
    if apply_detrend:
        detrended_data = np.zeros_like(processed_data)
        for i in range(processed_data.shape[0]):
            detrended_data[i, :] = detrend(processed_data[i, :], type='linear')
        processed_data = detrended_data
        print("  - Linear detrending applied")
    
    # Step 4c: Apply Gaussian filter
    if apply_gaussian:
        gaussian_data = np.zeros_like(processed_data)
        for i in range(processed_data.shape[0]):
            gaussian_data[i, :] = gaussian_filter1d(processed_data[i, :], sigma=gaussian_sigma)
        
        # Store non-gaussian data for plotting comparison
        non_gaussian_data = processed_data.copy()
        processed_data = gaussian_data
        print(f"  - Gaussian filtering applied (sigma={gaussian_sigma})")
        
        # Return Gaussian filtered data as main, non-gaussian as secondary
        return processed_data, non_gaussian_data
    else:
        # No Gaussian filtering
        return processed_data, None

def save_segment_data(segment_data, segment_name, toxin, trial_string, save_dir_data):
    """
    Step 5a: Save segment data to CSV
    """
    if segment_data is None:
        return None
    
    if save_dir_data is not None:
        data_path = Path(save_dir_data) / f"{segment_name}_{toxin}_{trial_string}.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df = pd.DataFrame(segment_data)
        df.to_csv(data_path, index=False)
        print(f"  Data saved to: {data_path}")
        return data_path
    
    return None

def plot_raw_and_filtered_segment(raw_segment, processed_segment, non_gaussian_segment, 
                                segment_name, toxin, trial_string, sampling_rate_hz=5, 
                                save_dir_plots=None, max_channels_plot=10):
    """
    Step 5b: Create plot showing raw and filtered data
    """
    if raw_segment is None:
        print(f"Cannot plot {segment_name} segment - data is None")
        return None
    
    n_channels, n_timepoints = raw_segment.shape
    time_axis = np.arange(n_timepoints) / sampling_rate_hz
    
    # Limit channels for clarity
    n_channels_plot = min(max_channels_plot, n_channels)
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
    ax.set_title(f'{segment_name.upper()} Segment - {toxin} {trial_string}\n(Processing Pipeline{gaussian_text})', 
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
        plot_path = Path(save_dir_plots) / f"{segment_name}_{toxin}_{trial_string}_processing.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {plot_path}")
    
    plt.close()  # Close to save memory
    return plot_path

def detect_voltage_events_fixed_threshold(data_segment, sampling_rate_hz=5, sigma_filter=3, threshold_multiplier=2.5, 
                                         apply_gaussian_for_detection=False, min_event_duration_sec=1.0, 
                                         fixed_threshold=0.15, use_fixed_threshold=True, **kwargs):
    """
    Event detection using FIXED threshold for completely consistent detection across all trials
    
    Parameters:
    -----------
    fixed_threshold : float
        Absolute threshold value (e.g., 0.15 for normalized [-1,1] data means 15% of full range)
    use_fixed_threshold : bool
        If True, uses fixed threshold. If False, falls back to global threshold approach
    """
    n_channels, n_timepoints = data_segment.shape
    events_list = []
    
    filtered_data = data_segment.copy()
    min_event_samples = int(min_event_duration_sec * sampling_rate_hz)
    
    if use_fixed_threshold:
        # FIXED THRESHOLD APPROACH: Same absolute threshold for ALL channels in ALL trials
        print(f"Using FIXED threshold event detection")
        print(f"Fixed threshold: ±{fixed_threshold:.3f}")
        print(f"Min duration: {min_event_duration_sec}s")
        
        # Calculate global baseline (should be ~0 for normalized data)
        global_mean = np.mean(filtered_data.flatten())
        
        # Use same fixed threshold for all channels
        channel_thresholds = [fixed_threshold] * n_channels
        channel_baselines = [global_mean] * n_channels
        
        print(f"  Global baseline: {global_mean:.4f}")
        print(f"  Fixed threshold applied to all {n_channels} channels: ±{fixed_threshold:.3f}")
        
    else:
        # Fallback to global threshold approach
        print(f"Using GLOBAL threshold event detection (fallback)")
        print(f"Threshold multiplier: {threshold_multiplier}σ")
        
        # Calculate global statistics
        all_data_flattened = filtered_data.flatten()
        global_mean = np.mean(all_data_flattened)
        global_std = np.std(all_data_flattened)
        global_threshold = threshold_multiplier * global_std
        
        channel_thresholds = [global_threshold] * n_channels
        channel_baselines = [global_mean] * n_channels
        
        print(f"  Global std: {global_std:.4f}, threshold: ±{global_threshold:.3f}")
    
    # Detect events for each channel using chosen threshold
    print(f"Detecting events using {'fixed' if use_fixed_threshold else 'global'} threshold...")
    
    for channel_idx in range(n_channels):
        channel_data = filtered_data[channel_idx, :]
        channel_threshold = channel_thresholds[channel_idx]
        baseline = channel_baselines[channel_idx]
        
        # Step 1: Find where signal exceeds threshold
        deviation_from_baseline = np.abs(channel_data - baseline)
        threshold_mask = deviation_from_baseline > channel_threshold
        
        # Apply morphological operations to clean up detection
        for _ in range(2):
            threshold_mask = binary_opening(threshold_mask)
        for _ in range(2):
            threshold_mask = binary_closing(threshold_mask)
        
        # Step 2: Find threshold crossing events
        threshold_starts = np.where(np.diff(threshold_mask.astype(int)) == 1)[0] + 1
        threshold_ends = np.where(np.diff(threshold_mask.astype(int)) == -1)[0] + 1
        
        # Handle edge cases
        if threshold_mask[0]:
            threshold_starts = np.concatenate([[0], threshold_starts])
        if threshold_mask[-1]:
            threshold_ends = np.concatenate([threshold_ends, [len(threshold_mask)-1]])
        
        # Step 3: For each threshold-crossing event, refine boundaries to baseline crossings
        for thresh_start, thresh_end in zip(threshold_starts, threshold_ends):
            
            # Skip very short threshold crossings first
            if thresh_end - thresh_start < min_event_samples // 2:
                continue
            
            # Determine event polarity from the threshold-crossing region
            event_region = channel_data[thresh_start:thresh_end+1]
            mean_deviation = np.mean(event_region - baseline)
            event_type = 'positive' if mean_deviation > 0 else 'negative'
            
            # Step 3a: Find event START by going backwards to baseline
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
            
            # Step 3b: Find event END by going forwards to baseline
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
            
            # Step 4: Apply minimum duration filter to refined event
            duration_samples = event_end - event_start + 1
            if duration_samples < min_event_samples:
                continue
            
            # Step 5: Calculate event properties using refined boundaries
            event_segment = channel_data[event_start:event_end+1]
            
            # Calculate amplitude (peak deviation from baseline)
            if event_type == 'positive':
                amplitude = np.max(event_segment) - baseline
                peak_value = np.max(event_segment)
            else:
                amplitude = baseline - np.min(event_segment)
                peak_value = np.min(event_segment)
            
            # Store event properties
            event_properties = {
                'channel_idx': channel_idx,
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
                'threshold_used': channel_threshold,
                'threshold_type': 'fixed' if use_fixed_threshold else 'global',
                'fixed_threshold_value': fixed_threshold if use_fixed_threshold else None,
                'threshold_start_sample': thresh_start,
                'threshold_end_sample': thresh_end
            }
            
            events_list.append(event_properties)
    
    threshold_type = 'fixed' if use_fixed_threshold else 'global'
    print(f"Detected {len(events_list)} events across {n_channels} channels using {threshold_type} threshold")
    
    # Return threshold info for plotting
    threshold_value = fixed_threshold if use_fixed_threshold else channel_thresholds[0]
    threshold_data = np.full((n_channels, n_timepoints), threshold_value)
    
    return events_list, filtered_data, threshold_data

def plot_events_with_arrows_fixed(raw_segment, processed_segment, events_list, 
                                segment_name, toxin, trial_string, sampling_rate_hz=5, 
                                save_dir_event_plots=None, max_channels_plot=10,
                                fixed_threshold=0.15, show_thresholds=True, use_fixed_threshold=True):
    """
    Plot timeseries with fixed threshold visualization (clean, no vertical lines)
    """
    if raw_segment is None:
        print(f"Cannot plot {segment_name} segment - data is None")
        return None
    
    n_channels, n_timepoints = raw_segment.shape
    time_axis = np.arange(n_timepoints) / sampling_rate_hz
    
    # Limit channels for clarity
    n_channels_plot = min(max_channels_plot, n_channels)
    n_channels_plot  = n_channels
    
    fig, ax = plt.subplots(figsize=(10,20))
    
    # Calculate channel spacing
    data_ranges = [np.ptp(raw_segment[i, :]) for i in range(n_channels_plot)]
    max_range = max(data_ranges) if data_ranges else 1
    channel_spacing = max_range * 1.5
    
    # Calculate baseline for visualization
    if show_thresholds and processed_segment is not None:
        if use_fixed_threshold:
            print(f"Using FIXED threshold for visualization: ±{fixed_threshold:.3f}")
            global_mean = np.mean(processed_segment.flatten())
            threshold_value = fixed_threshold
        else:
            print("Using GLOBAL threshold for visualization...")
            all_data_flattened = processed_segment.flatten()
            global_mean = np.mean(all_data_flattened)
            global_std = np.std(all_data_flattened)
            threshold_value = 2.5 * global_std  # fallback
    
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
        
        # Add threshold lines (straight horizontal lines)
        if show_thresholds and processed_segment is not None:
            # Use same baseline and threshold for all channels
            channel_mean = global_mean
            
            # Plot baseline (green)
            ax.axhline(y=channel_mean + offset, color='green', linestyle='-', 
                      linewidth=1.5, alpha=0.8, 
                      label='Fixed Baseline' if i == 0 else '')
            
            # Plot positive threshold (red)
            ax.axhline(y=channel_mean + threshold_value + offset, 
                      color='red', linestyle='--', linewidth=1, alpha=0.8,
                      label=f'Fixed +{threshold_value:.3f}' if i == 0 else '')
            
            # Plot negative threshold (blue)
            ax.axhline(y=channel_mean - threshold_value + offset, 
                      color='blue', linestyle='--', linewidth=1, alpha=0.8,
                      label=f'Fixed -{threshold_value:.3f}' if i == 0 else '')
        
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
            
            # NO VERTICAL LINES - clean visualization
    
    # Customize plot
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    threshold_type_text = "Fixed" if use_fixed_threshold else "Global"
    ax.set_title(f'{segment_name.upper()} Segment - {toxin} {trial_string}\n({threshold_type_text} Threshold Event Detection)', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, np.max(time_axis))
    ax.set_ylim(-channel_spacing*0.5, (n_channels_plot-0.5) * channel_spacing)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    if n_channels_plot > 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add event count text with threshold info
    total_events = len(events_list)
    positive_events = len([e for e in events_list if e['event_type'] == 'positive'])
    negative_events = len([e for e in events_list if e['event_type'] == 'negative'])
    
    if total_events > 0:
        avg_duration = np.mean([e['duration_sec'] for e in events_list])
        if use_fixed_threshold:
            event_text = f'Events: {total_events} total ({positive_events} pos, {negative_events} neg)\nFixed threshold: ±{fixed_threshold:.3f}\nAvg duration: {avg_duration:.2f}s'
        else:
            event_text = f'Events: {total_events} total ({positive_events} pos, {negative_events} neg)\nGlobal threshold\nAvg duration: {avg_duration:.2f}s'
    else:
        threshold_text = f"Fixed threshold: ±{fixed_threshold:.3f}" if use_fixed_threshold else "Global threshold"
        event_text = f'Events: 0 total\n{threshold_text}'
    
    ax.text(0.02, 0.98, event_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if save_dir_event_plots is not None:
        suffix = "fixed" if use_fixed_threshold else "global"
        plot_path = Path(save_dir_event_plots) / f"{segment_name}_{toxin}_{trial_string}_events_{suffix}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  {threshold_type_text} threshold plot saved to: {plot_path}")
    
    plt.close()
    return plot_path

def process_events_for_segments(segments_raw, segments_processed, toxin, trial_string, 
                               save_dir_events, save_dir_event_plots, sampling_rate_hz=5,
                               **event_detection_options):
    """
    Process both pre and post segments for event detection with fixed threshold
    """
    all_events = []
    
    for segment_name in ['pre', 'post']:
        raw_segment = segments_raw.get(segment_name)
        processed_segment = segments_processed.get(segment_name)
        
        if processed_segment is None:
            continue
            
        use_fixed = event_detection_options.get('use_fixed_threshold', True)
        threshold_type = "FIXED" if use_fixed else "GLOBAL"
        print(f"\n=== {threshold_type} EVENT DETECTION: {segment_name.upper()} SEGMENT ===")
        
        # Detect events using fixed threshold approach
        events_list, filtered_data, threshold_data = detect_voltage_events_fixed_threshold(
            processed_segment, 
            sampling_rate_hz=sampling_rate_hz,
            **event_detection_options
        )
        
        # Add metadata to events
        for event in events_list:
            event.update({
                'trial_string': trial_string,
                'toxin': toxin,
                'segment': segment_name,
                'cell_index': event['channel_idx']
            })
        
        all_events.extend(events_list)
        
        # Create plot with fixed threshold visualization
        plot_events_with_arrows_fixed(
            raw_segment, processed_segment, events_list, segment_name,
            toxin, trial_string, sampling_rate_hz, save_dir_event_plots,
            fixed_threshold=event_detection_options.get('fixed_threshold', 0.15),
            show_thresholds=True,
            use_fixed_threshold=use_fixed
        )
    
    # Save events to CSV (updated column order)
    if all_events:
        events_df = pd.DataFrame(all_events)
        
        column_order = ['trial_string', 'toxin', 'segment', 'cell_index', 
                       'start_time_sec', 'end_time_sec', 'duration_sec', 
                       'amplitude', 'event_type', 'mean_amplitude', 'peak_value',
                       'baseline_value', 'threshold_used', 'threshold_type', 'fixed_threshold_value',
                       'start_sample', 'end_sample', 'duration_samples',
                       'threshold_start_sample', 'threshold_end_sample']
        
        events_df = events_df[column_order]
        
        # Save events
        threshold_suffix = "fixed" if use_fixed else "global"
        events_path = Path(save_dir_events) / f"events_{toxin}_{trial_string}_{threshold_suffix}.csv"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(events_path, index=False)
        print(f"Events saved to: {events_path}")
        
        # Print summary with threshold info
        durations = events_df['duration_sec'].values
        print(f"\n{threshold_type} Event Summary for {toxin} {trial_string}:")
        print(f"  Total events: {len(all_events)}")
        print(f"  Pre-segment events: {len([e for e in all_events if e['segment'] == 'pre'])}")
        print(f"  Post-segment events: {len([e for e in all_events if e['segment'] == 'post'])}")
        if len(durations) > 0:
            print(f"  Duration stats: mean={np.mean(durations):.2f}s, median={np.median(durations):.2f}s")
        
        # Print threshold info
        if use_fixed:
            fixed_threshold_val = events_df['fixed_threshold_value'].iloc[0] if len(events_df) > 0 else None
            if fixed_threshold_val:
                print(f"  Fixed threshold used: ±{fixed_threshold_val:.3f}")
        
        return events_df
    
    return None

def combine_all_events(save_dir_events, output_filename="all_events_combined_fixed.csv"):
    """
    Combine all individual event CSV files into one master file
    """
    events_dir = Path(save_dir_events)
    event_files = list(events_dir.glob("events_*_fixed.csv"))
    
    if not event_files:
        print("No fixed threshold event files found to combine")
        return None
    
    all_dataframes = []
    for file in event_files:
        df = pd.read_csv(file)
        all_dataframes.append(df)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save combined file
    combined_path = events_dir / output_filename
    combined_df.to_csv(combined_path, index=False)
    
    print(f"\nCombined events file saved: {combined_path}")
    print(f"Total events across all experiments: {len(combined_df)}")
    
    # Print summary statistics
    print("\nSummary by toxin:")
    print(combined_df.groupby(['toxin', 'segment']).size().unstack(fill_value=0))
    
    # Print fixed threshold info
    if 'fixed_threshold_value' in combined_df.columns:
        unique_thresholds = combined_df['fixed_threshold_value'].dropna().unique()
        print(f"\nFixed threshold(s) used: {unique_thresholds}")
    
    return combined_df

def process_single_file_complete(csv_file, toxin, save_dir_plots, save_dir_data, save_dir_events, save_dir_event_plots,
                               apply_mean_center=True, apply_detrend=True, 
                               apply_gaussian=False, gaussian_sigma=3,
                               apply_normalization=True,
                               enable_event_detection=True, **event_options):
    """
    Complete flexible workflow with early normalization and fixed threshold event detection
    """
    print(f"\nProcessing: {csv_file.name}")
    print("-" * 50)
    
    try:
        # Extract trial info
        trial = csv_file.stem
        trial_string = '_'.join(trial.split('_')[:3])
        
        print(f"Trial: {trial}")
        print(f"Trial string: {trial_string}")
        
        # Load data
        data_matrix = pd.read_csv(csv_file)
        data_matrix = data_matrix.iloc[:, -10000:]
        
        # Step 1: Apply normalization + high-pass filter to full timeseries
        highpass_data = apply_highpass_to_full_timeseries(
            data_matrix, 
            sampling_rate_hz=5, 
            cutoff_freq=0.01,
            apply_normalization=apply_normalization
        )
        
        # Step 2: Find consensus timepoint
        consensus_timepoint = find_consensus_timepoint_only(
            highpass_data,
            sampling_rate_hz=5,
            time_window=(4500, 5500),
            detection_method='zscore',
            consensus_method='mode'
        )
        
        if consensus_timepoint is None:
            print("✗ WARNING: No consensus found - cannot segment")
            return False
        
        samples_per_min = 5 * 60
        print(f"✓ SUCCESS: Consensus at {consensus_timepoint/samples_per_min:.2f} min")
        
        # Step 3: Segment the normalized + high-pass filtered timeseries
        segments = segment_highpass_timeseries(highpass_data, consensus_timepoint)
        
        # Store both raw and processed segments for event detection
        segments_raw = segments  # Normalized + high-pass filtered segments
        segments_processed = {}
        
        # Step 4 & 5: Process each segment
        for segment_name in ['pre', 'post']:
            if segments[segment_name] is not None:
                print(f"\n=== PROCESSING {segment_name.upper()} SEGMENT ===")
                
                # Step 4: Apply processing (mean center + detrend + optional gaussian)
                processed_data, non_gaussian_data = apply_segment_processing(
                    segments[segment_name],
                    apply_mean_center=apply_mean_center,
                    apply_detrend=apply_detrend,
                    apply_gaussian=apply_gaussian,
                    gaussian_sigma=gaussian_sigma
                )
                
                segments_processed[segment_name] = processed_data
                
                # Step 5a: Save processed data
                save_segment_data(processed_data, segment_name, toxin, trial_string, save_dir_data)
                
                # Step 5b: Create comparison plot
                plot_raw_and_filtered_segment(
                    segments[segment_name], processed_data, non_gaussian_data,
                    segment_name, toxin, trial_string, 
                    save_dir_plots=save_dir_plots
                )
        
        # Step 6: Event detection with fixed threshold (optional)
        if enable_event_detection:
            print(f"\n=== STEP 6: FIXED THRESHOLD EVENT DETECTION ===")
            events_df = process_events_for_segments(
                segments_raw, segments_processed, toxin, trial_string,
                save_dir_events, save_dir_event_plots,
                sampling_rate_hz=5, **event_options
            )
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR processing {csv_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main workflow with early normalization and fixed threshold event detection
    """
    # Set up directories
    home = Path.home()
    if "ys5320" in str(home):
        top_dir = Path(home, "firefly_link")
    else:
        top_dir = Path(r'R:\home\firefly_link')
    
    save_dir = Path(top_dir, r'Calcium_Voltage_Imaging\code_yilin\infusion_detection_check\fixed_std')
    data_dir = Path(top_dir, r'ca_voltage_imaging_working\results')
    save_dir_data = Path(save_dir, r'tc_fixed')
    save_dir_plots = Path(save_dir, r'highpass_segmented_processing_fixed')
    save_dir_events = Path(save_dir, r'events_fixed')
    save_dir_event_plots = Path(save_dir, r'tc_with_events_fixed')
    
    # Create directories
    save_dir_data.mkdir(parents=True, exist_ok=True)
    save_dir_plots.mkdir(parents=True, exist_ok=True)
    save_dir_events.mkdir(parents=True, exist_ok=True)
    save_dir_event_plots.mkdir(parents=True, exist_ok=True)
    
    # Processing options - normalization moved to very beginning
    PROCESSING_OPTIONS = {
        'apply_mean_center': True,
        'apply_detrend': True,
        'apply_gaussian': True,
        'gaussian_sigma': 3,
        'apply_normalization': True     # Applies at the very beginning!
    }
    
    # Event detection options - FIXED THRESHOLD
    EVENT_DETECTION_OPTIONS = {
        'enable_event_detection': True,
        'fixed_threshold': 0.2,                 #
        'use_fixed_threshold': True,             # Use fixed threshold instead of statistical
        'min_event_duration_sec': 1.5,
        'apply_gaussian_for_detection': False,
    }
    
    print("=== FIXED THRESHOLD WORKFLOW ===")
    print("Raw data → NORMALIZE [-1,1] → High-pass → Segment → Mean center → Detrend → Gaussian → FIXED Event Detection")
    print("🎯 FIXED THRESHOLD: Same absolute threshold for ALL channels in ALL trials")
    print(f"🔧 Threshold value: ±{EVENT_DETECTION_OPTIONS['fixed_threshold']:.3f} (adjust this to tune sensitivity)")
    print()
     
    toxins = ['L-15_control', 'Ani9_10uM', 'ATP_1mM', 'TRAM-34_1uM', 'dantrolene_10uM']
    toxins = ['ATP_1mM', 'TRAM-34_1uM'] 
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

    
    for toxin in toxins:
        print(f"\n{'='*60}")
        print(f"PROCESSING TOXIN: {toxin}")
        print(f"{'='*60}")
        
        # Find matching files
        all_csv_files = list(data_dir.glob('*_voltage_transfected*.csv'))
        csv_files = []
        
        for file in all_csv_files:
            if toxin.lower() == 'dantrolene_10um':
                if 'dantrolene_10um' in file.name.lower():
                    csv_files.append(file)
            else:
                if toxin.lower() in file.name.lower():
                    csv_files.append(file)
        
        print(f"Found {len(csv_files)} files for {toxin}:")
        for file in csv_files:
            print(f"  - {file.name}")
        
        if not csv_files:
            print("No files found!")
            continue
        
        # Process each file
        for csv_file in csv_files:
            total_processed += 1
            success = process_single_file_complete(
                csv_file, toxin, save_dir_plots, save_dir_data, save_dir_events, save_dir_event_plots,
                **PROCESSING_OPTIONS, **EVENT_DETECTION_OPTIONS
            )
            if success:
                total_successful += 1
            print("-" * 50)
    
    # After processing all files, combine events if event detection was enabled
    if EVENT_DETECTION_OPTIONS['enable_event_detection']:
        print(f"\n{'='*60}")
        print("COMBINING ALL FIXED THRESHOLD EVENTS")
        print(f"{'='*60}")
        combined_events = combine_all_events(save_dir_events)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FIXED THRESHOLD PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {total_processed}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_processed - total_successful}")
    print(f"Data saved to: {save_dir_data}")
    print(f"Processing plots saved to: {save_dir_plots}")
    if EVENT_DETECTION_OPTIONS['enable_event_detection']:
        print(f"Event data saved to: {save_dir_events}")
        print(f"Event plots saved to: {save_dir_event_plots}")
    
    print("\n=== FIXED THRESHOLD SUMMARY ===")
    print("✅ Early normalization: All data normalized to [-1,1] before processing")
    print(f"✅ Fixed threshold: ±{EVENT_DETECTION_OPTIONS['fixed_threshold']:.3f} applied to ALL channels in ALL trials")
    print("✅ Expected result: Perfect consistency across all trials!")
    print("✅ Expected result: No bias between quiet and active trials!")
    print("✅ Expected result: Same detection sensitivity everywhere!")


if __name__ == "__main__":
    main()