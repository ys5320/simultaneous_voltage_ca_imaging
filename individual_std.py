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
    
    # NEW: Apply normalization FIRST (before any filtering)
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
    Step 4: Apply processing to individual segments (no normalization here anymore)
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
    Now: processed_segment is the FINAL data used (Gaussian filtered if enabled)
          non_gaussian_segment is the intermediate step (mean-centered + detrended only)
    """
    if raw_segment is None:
        print(f"Cannot plot {segment_name} segment - data is None")
        return None
    
    n_channels, n_timepoints = raw_segment.shape
    time_axis = np.arange(n_timepoints) / sampling_rate_hz
    
    # Limit channels for clarity
    n_channels_plot = min(max_channels_plot, n_channels)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate channel spacing
    data_ranges = [np.ptp(raw_segment[i, :]) for i in range(n_channels_plot)]
    max_range = max(data_ranges) if data_ranges else 1
    channel_spacing = max_range * 0.5
    
    for i in range(n_channels_plot):
        offset = i * channel_spacing
        
        # Plot raw data (grey, thinner line)
        ax.plot(time_axis, raw_segment[i, :] + offset, color='grey', 
               linewidth=0.8, alpha=0.6, label='High-pass only' if i == 0 else '')
        
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

def detect_voltage_events_optimized(data_segment, sampling_rate_hz=5, sigma_filter=3, threshold_multiplier=2.5, 
                                   apply_gaussian_for_detection=False, min_event_duration_sec=1.0, **kwargs):
    """
    Optimized voltage event detection with baseline boundary detection
    
    Changes:
    1. Events are detected when signal exceeds threshold
    2. Event boundaries (start/end) are refined to baseline crossings
    3. Increased default min_event_duration_sec to filter noise
    """
    n_channels, n_timepoints = data_segment.shape
    events_list = []
    
    # Use data as-is (already processed)
    print(f"Using already processed data for event detection (min duration: {min_event_duration_sec}s)")
    filtered_data = data_segment.copy()
    
    # Calculate STATIC standard deviation for each channel (from entire timeseries)
    print("Calculating static thresholds for event detection...")
    static_std = np.zeros(n_channels)
    
    for i in range(n_channels):
        static_std[i] = np.std(data_segment[i, :])
    
    # Detect events for each channel using static threshold
    print("Detecting events with optimized boundaries...")
    min_event_samples = int(min_event_duration_sec * sampling_rate_hz)
    
    for channel_idx in range(n_channels):
        channel_data = filtered_data[channel_idx, :]
        channel_static_std = static_std[channel_idx]
        baseline = np.mean(channel_data)  # Should be ~0 since mean-centered
        
        # Static threshold (same for all timepoints)
        static_threshold = threshold_multiplier * channel_static_std
        
        # Step 1: Find where signal exceeds threshold (detection criterion)
        deviation_from_baseline = np.abs(channel_data - baseline)
        threshold_mask = deviation_from_baseline > static_threshold
        
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
            if thresh_end - thresh_start < min_event_samples // 2:  # Use half min duration for initial filter
                continue
            
            # Determine event polarity from the threshold-crossing region
            event_region = channel_data[thresh_start:thresh_end+1]
            mean_deviation = np.mean(event_region - baseline)
            event_type = 'positive' if mean_deviation > 0 else 'negative'
            
            # Step 3a: Find event START by going backwards from threshold crossing to baseline
            event_start = thresh_start
            for i in range(thresh_start - 1, -1, -1):
                if event_type == 'positive':
                    # For positive events, find where signal goes back to or below baseline
                    if channel_data[i] <= baseline:
                        event_start = i + 1  # Event starts after baseline crossing
                        break
                else:
                    # For negative events, find where signal goes back to or above baseline  
                    if channel_data[i] >= baseline:
                        event_start = i + 1  # Event starts after baseline crossing
                        break
                event_start = i  # Keep going back if we haven't found baseline yet
            
            # Step 3b: Find event END by going forwards from threshold crossing to baseline
            event_end = thresh_end
            for i in range(thresh_end + 1, len(channel_data)):
                if event_type == 'positive':
                    # For positive events, find where signal returns to or below baseline
                    if channel_data[i] <= baseline:
                        event_end = i - 1  # Event ends before baseline crossing
                        break
                else:
                    # For negative events, find where signal returns to or above baseline
                    if channel_data[i] >= baseline:
                        event_end = i - 1  # Event ends before baseline crossing
                        break
                event_end = i  # Keep going forward if we haven't found baseline yet
            
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
                'static_threshold_used': static_threshold,
                'threshold_start_sample': thresh_start,  # Original threshold crossing for reference
                'threshold_end_sample': thresh_end
            }
            
            events_list.append(event_properties)
    
    print(f"Detected {len(events_list)} events across {n_channels} channels using optimized detection")
    print(f"  Minimum event duration: {min_event_duration_sec}s ({min_event_samples} samples)")
    
    # Return static std data for plotting (constant across time)
    std_data_for_plotting = np.tile(static_std[:, np.newaxis], (1, n_timepoints))
    
    return events_list, filtered_data, std_data_for_plotting

def plot_events_with_arrows_optimized(raw_segment, processed_segment, events_list, 
                                     segment_name, toxin, trial_string, sampling_rate_hz=5, 
                                     save_dir_event_plots=None, max_channels_plot=10,
                                     threshold_multiplier=2.5, show_thresholds=True):
    """
    Plot timeseries with arrows showing optimized event boundaries
    """
    if raw_segment is None:
        print(f"Cannot plot {segment_name} segment - data is None")
        return None
    
    n_channels, n_timepoints = raw_segment.shape
    time_axis = np.arange(n_timepoints) / sampling_rate_hz
    
    # Limit channels for clarity
    #n_channels_plot = min(max_channels_plot, n_channels)
    n_channels_plot = n_channels
    fig, ax = plt.subplots(figsize=(10,20))
    
    # Calculate channel spacing
    data_ranges = [np.ptp(raw_segment[i, :]) for i in range(n_channels_plot)]
    max_range = max(data_ranges) if data_ranges else 1
    channel_spacing = max_range * 0.5
    
    # Calculate STATIC thresholds for each channel if needed
    channel_thresholds = {}
    if show_thresholds and processed_segment is not None:
        print("Calculating STATIC threshold lines for visualization...")
        
        for i in range(n_channels_plot):
            channel_data = processed_segment[i, :]
            
            # Calculate STATIC statistics for entire timeseries
            channel_mean = np.mean(channel_data)  # Should be ~0 since mean-centered
            channel_std = np.std(channel_data)    # Global standard deviation
            
            # Create constant threshold arrays (straight lines)
            mean_array = np.full_like(time_axis, channel_mean)
            positive_threshold_array = np.full_like(time_axis, channel_mean + threshold_multiplier * channel_std)
            negative_threshold_array = np.full_like(time_axis, channel_mean - threshold_multiplier * channel_std)
            
            channel_thresholds[i] = {
                'mean': mean_array,
                'positive_threshold': positive_threshold_array,
                'negative_threshold': negative_threshold_array,
                'static_std': channel_std,
                'static_threshold_value': threshold_multiplier * channel_std
            }
    
    for i in range(n_channels_plot):
        offset = i * channel_spacing
        
        # Plot raw data (grey) and processed data (black)
        ax.plot(time_axis, raw_segment[i, :] + offset, color='grey', 
               linewidth=0.8, alpha=0.7, label='High-pass only' if i == 0 else '')
        
        if processed_segment is not None:
            ax.plot(time_axis, processed_segment[i, :] + offset, color='black', 
                   linewidth=1.2, alpha=0.9, label='Processed (Gaussian filtered)' if i == 0 else '')
        
        # Add channel label
        ax.text(-0.02 * np.max(time_axis), np.mean(raw_segment[i, :]) + offset, f'{i}', 
                verticalalignment='center', fontsize=9, fontweight='bold')
        
        # Add STATIC threshold lines if requested
        if show_thresholds and i in channel_thresholds:
            thresholds = channel_thresholds[i]
            
            # Plot mean line (baseline - should be around 0)
            ax.plot(time_axis, thresholds['mean'] + offset, 
                   color='green', linestyle='-', linewidth=1.5, alpha=0.8,
                   label='Baseline (mean)' if i == 0 else '')
            
            # Plot positive threshold line (STRAIGHT LINE)
            ax.plot(time_axis, thresholds['positive_threshold'] + offset, 
                   color='red', linestyle='--', linewidth=1, alpha=0.8,
                   label=f'+{threshold_multiplier}σ threshold' if i == 0 else '')
            
            # Plot negative threshold line (STRAIGHT LINE)
            ax.plot(time_axis, thresholds['negative_threshold'] + offset, 
                   color='blue', linestyle='--', linewidth=1, alpha=0.8,
                   label=f'-{threshold_multiplier}σ threshold' if i == 0 else '')
        
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
                   color=event_color, linewidth=3, alpha=0.7,
                   label=f'{event["event_type"].title()} events' if i == 0 and 
                         not any(event_color in str(h.get_color()) for h in ax.get_children()) else '')
            
            # Add start and end markers
            ax.plot(event_start_time, y_start, 'o', color=event_color, markersize=4, alpha=0.9)
            ax.plot(event_end_time, y_end, 's', color=event_color, markersize=4, alpha=0.9)
            
            # Add vertical lines showing event boundaries
            #ax.axvline(x=event_start_time, color=event_color, linestyle=':', alpha=0.5, linewidth=1)
            #ax.axvline(x=event_end_time, color=event_color, linestyle=':', alpha=0.5, linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(f'{segment_name.upper()} Segment - {toxin} {trial_string}\n(Optimized Event Detection: Baseline Boundaries)', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, np.max(time_axis))
    ax.set_ylim(-channel_spacing*0.5, (n_channels_plot-0.5) * channel_spacing)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    if n_channels_plot > 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add event count text with duration info
    total_events = len(events_list)
    positive_events = len([e for e in events_list if e['event_type'] == 'positive'])
    negative_events = len([e for e in events_list if e['event_type'] == 'negative'])
    
    if total_events > 0:
        avg_duration = np.mean([e['duration_sec'] for e in events_list])
        event_text = f'Events: {total_events} total ({positive_events} pos, {negative_events} neg)\nAvg duration: {avg_duration:.2f}s'
    else:
        event_text = 'Events: 0 total'
    
    ax.text(0.02, 0.98, event_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot to the event plots directory
    if save_dir_event_plots is not None:
        plot_path = Path(save_dir_event_plots) / f"{segment_name}_{toxin}_{trial_string}_events_optimized.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Optimized event plot saved to: {plot_path}")
    
    plt.close()
    return plot_path

def detect_voltage_events(data_segment, sampling_rate_hz=5, sigma_filter=3, threshold_multiplier=2.5, 
                         apply_gaussian_for_detection=False, min_event_duration_sec=0.5, **kwargs):
    """
    Detect voltage fluctuation events using STATIC thresholds
    """
    n_channels, n_timepoints = data_segment.shape
    events_list = []
    
    # Use data as-is (already processed)
    print("Using already processed data for event detection")
    filtered_data = data_segment.copy()
    
    # Calculate STATIC standard deviation for each channel (from entire timeseries)
    print("Calculating static thresholds for event detection...")
    static_std = np.zeros(n_channels)
    
    for i in range(n_channels):
        # Calculate global std for this channel
        static_std[i] = np.std(data_segment[i, :])
    
    # Detect events for each channel using static threshold
    print("Detecting events with static thresholds...")
    min_event_samples = int(min_event_duration_sec * sampling_rate_hz)
    
    for channel_idx in range(n_channels):
        channel_data = filtered_data[channel_idx, :]
        channel_static_std = static_std[channel_idx]
        
        # Static threshold (same for all timepoints)
        static_threshold = threshold_multiplier * channel_static_std
        
        # Detect events: when signal diverges from 0 (baseline) by more than static threshold
        deviation_from_baseline = np.abs(channel_data)  # Since data is mean-centered around 0
        
        # Create binary event mask using static threshold
        event_mask = deviation_from_baseline > static_threshold
        
        # Apply morphological operations (as mentioned in paper)
        # 2 iterations of binary opening (removes short events)
        for _ in range(2):
            event_mask = binary_opening(event_mask)
        
        # 2 rounds of binary closing (merges neighboring events)
        for _ in range(2):
            event_mask = binary_closing(event_mask)
        
        # Find event boundaries
        event_starts = np.where(np.diff(event_mask.astype(int)) == 1)[0] + 1
        event_ends = np.where(np.diff(event_mask.astype(int)) == -1)[0] + 1
        
        # Handle edge cases
        if event_mask[0]:
            event_starts = np.concatenate([[0], event_starts])
        if event_mask[-1]:
            event_ends = np.concatenate([event_ends, [len(event_mask)-1]])
        
        # Process each event
        for start_idx, end_idx in zip(event_starts, event_ends):
            duration_samples = end_idx - start_idx + 1
            
            # Skip events shorter than minimum duration
            if duration_samples < min_event_samples:
                continue
            
            # Calculate event properties
            event_segment = channel_data[start_idx:end_idx+1]
            baseline_value = 0  # Since data is mean-centered around 0
            
            # Determine if positive or negative going
            mean_deviation = np.mean(event_segment - baseline_value)
            event_type = 'positive' if mean_deviation > 0 else 'negative'
            
            # Calculate amplitude (peak deviation from baseline)
            if event_type == 'positive':
                amplitude = np.max(event_segment) - baseline_value
            else:
                amplitude = baseline_value - np.min(event_segment)
            
            # Store event properties
            event_properties = {
                'channel_idx': channel_idx,
                'start_time_sec': start_idx / sampling_rate_hz,
                'end_time_sec': end_idx / sampling_rate_hz,
                'start_sample': start_idx,
                'end_sample': end_idx,
                'duration_sec': duration_samples / sampling_rate_hz,
                'duration_samples': duration_samples,
                'amplitude': amplitude,
                'event_type': event_type,
                'mean_amplitude': np.abs(mean_deviation),
                'peak_value': np.max(event_segment) if event_type == 'positive' else np.min(event_segment),
                'static_threshold_used': static_threshold
            }
            
            events_list.append(event_properties)
    
    print(f"Detected {len(events_list)} events across {n_channels} channels using static thresholds")
    
    # Return static std data for plotting (constant across time)
    std_data_for_plotting = np.tile(static_std[:, np.newaxis], (1, n_timepoints))
    
    return events_list, filtered_data, std_data_for_plotting

def plot_events_with_arrows(raw_segment, processed_segment, events_list, 
                           segment_name, toxin, trial_string, sampling_rate_hz=5, 
                           save_dir_event_plots=None, max_channels_plot=10,
                           threshold_multiplier=2.5, show_thresholds=True):
    """
    Plot timeseries with arrows pointing to detected events using STATIC thresholds
    """
    if raw_segment is None:
        print(f"Cannot plot {segment_name} segment - data is None")
        return None
    
    n_channels, n_timepoints = raw_segment.shape
    time_axis = np.arange(n_timepoints) / sampling_rate_hz
    
    # Limit channels for clarity
    #n_channels_plot = min(max_channels_plot, n_channels)
    n_channels_plot = n_channels
    
    fig, ax = plt.subplots(figsize=(10,20))
    
    # Calculate channel spacing
    data_ranges = [np.ptp(raw_segment[i, :]) for i in range(n_channels_plot)]
    max_range = max(data_ranges) if data_ranges else 1
    channel_spacing = max_range * 0.5
    
    # Calculate STATIC thresholds for each channel if needed
    channel_thresholds = {}
    if show_thresholds and processed_segment is not None:
        print("Calculating STATIC threshold lines for visualization...")
        
        for i in range(n_channels_plot):
            channel_data = processed_segment[i, :]
            
            # Calculate STATIC statistics for entire timeseries
            channel_mean = np.mean(channel_data)  # Should be ~0 since mean-centered
            channel_std = np.std(channel_data)    # Global standard deviation
            
            # Create constant threshold arrays (straight lines)
            mean_array = np.full_like(time_axis, channel_mean)
            positive_threshold_array = np.full_like(time_axis, channel_mean + threshold_multiplier * channel_std)
            negative_threshold_array = np.full_like(time_axis, channel_mean - threshold_multiplier * channel_std)
            
            channel_thresholds[i] = {
                'mean': mean_array,
                'positive_threshold': positive_threshold_array,
                'negative_threshold': negative_threshold_array,
                'static_std': channel_std,
                'static_threshold_value': threshold_multiplier * channel_std
            }
    
    for i in range(n_channels_plot):
        offset = i * channel_spacing
        
        # Plot raw data (grey) and processed data (black)
        ax.plot(time_axis, raw_segment[i, :] + offset, color='grey', 
               linewidth=0.8, alpha=0.7, label='High-pass only' if i == 0 else '')
        
        if processed_segment is not None:
            ax.plot(time_axis, processed_segment[i, :] + offset, color='black', 
                   linewidth=1.2, alpha=0.9, label='Processed (Gaussian filtered)' if i == 0 else '')
        
        # Add channel label
        ax.text(-0.02 * np.max(time_axis), np.mean(raw_segment[i, :]) + offset, f'{i}', 
                verticalalignment='center', fontsize=9, fontweight='bold')
        
        # Add STATIC threshold lines if requested
        if show_thresholds and i in channel_thresholds:
            thresholds = channel_thresholds[i]
            
            # Plot mean line (baseline - should be around 0)
            ax.plot(time_axis, thresholds['mean'] + offset, 
                   color='green', linestyle='-', linewidth=1, alpha=0.6,
                   label='Mean (baseline)' if i == 0 else '')
            
            # Plot positive threshold line (STRAIGHT LINE)
            ax.plot(time_axis, thresholds['positive_threshold'] + offset, 
                   color='red', linestyle='--', linewidth=1, alpha=0.8,
                   label=f'+{threshold_multiplier}σ threshold (static)' if i == 0 else '')
            
            # Plot negative threshold line (STRAIGHT LINE)
            ax.plot(time_axis, thresholds['negative_threshold'] + offset, 
                   color='blue', linestyle='--', linewidth=1, alpha=0.8,
                   label=f'-{threshold_multiplier}σ threshold (static)' if i == 0 else '')
            
            # Add text annotation showing the static threshold value
            if i == 0:  # Only for first channel to avoid clutter
                threshold_val = thresholds['static_threshold_value']
                ax.text(0.98, 0.02, f'Static threshold: ±{threshold_val:.3f}', 
                       transform=ax.transAxes, fontsize=9, ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add event arrows for this channel
        channel_events = [event for event in events_list if event['channel_idx'] == i]
        
        for event in channel_events:
            event_start_time = event['start_time_sec']
            
            # Get y-position for arrow (use processed data if available)
            if processed_segment is not None:
                y_pos = processed_segment[i, event['start_sample']] + offset
            else:
                y_pos = raw_segment[i, event['start_sample']] + offset
            
            # Add arrow pointing to event start
            arrow_color = 'red' if event['event_type'] == 'positive' else 'blue'
            ax.annotate('', xy=(event_start_time, y_pos), 
                       xytext=(event_start_time, y_pos + channel_spacing * 0.3),
                       arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    
    # Customize plot
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(f'{segment_name.upper()} Segment - {toxin} {trial_string} (Static Thresholds, Gaussian Filtered)', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, np.max(time_axis))
    ax.set_ylim(-channel_spacing*0.5, (n_channels_plot-0.5) * channel_spacing)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    if n_channels_plot > 0:
        ax.legend(loc='upper right')
    
    # Add event count text
    total_events = len(events_list)
    positive_events = len([e for e in events_list if e['event_type'] == 'positive'])
    negative_events = len([e for e in events_list if e['event_type'] == 'negative'])
    
    event_text = f'Events: {total_events} total ({positive_events} pos, {negative_events} neg)'
    ax.text(0.02, 0.98, event_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot to the event plots directory
    if save_dir_event_plots is not None:
        plot_path = Path(save_dir_event_plots) / f"{segment_name}_{toxin}_{trial_string}_events_static.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Event plot saved to: {plot_path}")
    
    plt.close()
    return plot_path

def process_events_for_segments(segments_raw, segments_processed, toxin, trial_string, 
                               save_dir_events, save_dir_event_plots, sampling_rate_hz=5,
                               **event_detection_options):
    """
    Process both pre and post segments for event detection with optimization
    """
    all_events = []
    
    for segment_name in ['pre', 'post']:
        raw_segment = segments_raw.get(segment_name)
        processed_segment = segments_processed.get(segment_name)
        
        if processed_segment is None:
            continue
            
        print(f"\n=== OPTIMIZED EVENT DETECTION: {segment_name.upper()} SEGMENT ===")
        
        # Detect events using OPTIMIZED function
        events_list, filtered_data, std_data = detect_voltage_events_optimized(
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
                'cell_index': event['channel_idx']  # Rename for clarity
            })
        
        all_events.extend(events_list)
        
        # Create plot with OPTIMIZED visualization
        plot_events_with_arrows_optimized(
            raw_segment, processed_segment, events_list, segment_name,
            toxin, trial_string, sampling_rate_hz, save_dir_event_plots,
            threshold_multiplier=event_detection_options.get('threshold_multiplier', 2.5),
            show_thresholds=True
        )
    
    # Save events to CSV (same as before)
    if all_events:
        events_df = pd.DataFrame(all_events)
        
        # Updated column order to include new fields
        column_order = ['trial_string', 'toxin', 'segment', 'cell_index', 
                       'start_time_sec', 'end_time_sec', 'duration_sec', 
                       'amplitude', 'event_type', 'mean_amplitude', 'peak_value',
                       'baseline_value', 'static_threshold_used',
                       'start_sample', 'end_sample', 'duration_samples',
                       'threshold_start_sample', 'threshold_end_sample']
        
        events_df = events_df[column_order]
        
        # Save events
        events_path = Path(save_dir_events) / f"events_{toxin}_{trial_string}_optimized.csv"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(events_path, index=False)
        print(f"Events saved to: {events_path}")
        
        # Print summary with duration statistics
        durations = events_df['duration_sec'].values
        print(f"\nEvent Summary for {toxin} {trial_string}:")
        print(f"  Total events: {len(all_events)}")
        print(f"  Pre-segment events: {len([e for e in all_events if e['segment'] == 'pre'])}")
        print(f"  Post-segment events: {len([e for e in all_events if e['segment'] == 'post'])}")
        if len(durations) > 0:
            print(f"  Duration stats: mean={np.mean(durations):.2f}s, median={np.median(durations):.2f}s, range={np.min(durations):.2f}-{np.max(durations):.2f}s")
        
        return events_df
    
    return None

def combine_all_events(save_dir_events, output_filename="all_events_combined.csv"):
    """
    Combine all individual event CSV files into one master file
    """
    events_dir = Path(save_dir_events)
    event_files = list(events_dir.glob("events_*.csv"))
    
    if not event_files:
        print("No event files found to combine")
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
    
    return combined_df

def process_single_file_complete(csv_file, toxin, save_dir_plots, save_dir_data, save_dir_events, save_dir_event_plots,
                               apply_mean_center=True, apply_detrend=True, 
                               apply_gaussian=False, gaussian_sigma=3,
                               apply_normalization=True,  # Now controls early normalization
                               enable_event_detection=True, **event_options):
    """
    Complete flexible workflow with early normalization
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
            apply_normalization=apply_normalization  # Pass normalization flag
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
                # No normalization here since it's already done
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
        
        # Step 6: Event detection (optional)
        if enable_event_detection:
            print(f"\n=== STEP 6: EVENT DETECTION ===")
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
    Main workflow with early normalization for consistent scaling
    """
    # Set up directories
    home = Path.home()
    if "ys5320" in str(home):
        top_dir = Path(home, "firefly_link")
    else:
        top_dir = Path(r'R:\home\firefly_link')
    
    data_dir = Path(top_dir, r'ca_voltage_imaging_working\results')
    save_dir_data = Path(top_dir, r'Calcium_Voltage_Imaging\code_yilin\infusion_detection_check\tc')
    save_dir_plots = Path(top_dir, r'Calcium_Voltage_Imaging\code_yilin\infusion_detection_check\highpass_segmented_processing')
    save_dir_events = Path(top_dir, r'Calcium_Voltage_Imaging\code_yilin\infusion_detection_check\events')
    save_dir_event_plots = Path(top_dir, r'Calcium_Voltage_Imaging\code_yilin\infusion_detection_check\tc_with_events')
    
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
        'apply_normalization': True     # Now applies at the very beginning!
    }
    
    # Event detection options - keep the same 2.5σ threshold
    EVENT_DETECTION_OPTIONS = {
        'enable_event_detection': True,
        'threshold_multiplier': 2.5,            # Same 2.5σ threshold works great on normalized data
        'apply_gaussian_for_detection': False,
        'min_event_duration_sec': 1.5,
    }
    
    print("=== UPDATED WORKFLOW SUMMARY ===")
    print("Raw data → NORMALIZE [-1,1] → High-pass → Segment → Mean center → Detrend → Gaussian → Event Detection (2.5σ)")
    print("Early normalization ensures consistent scaling throughout the entire pipeline")
    print()
    
    #toxins = ['L-15_control','Ani9_10uM','ATP_1mM','TRAM-34_1uM','dantrolene_10uM']
    toxins = ['ATP_1mM','TRAM-34_1uM']
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
        print("COMBINING ALL EVENTS")
        print(f"{'='*60}")
        combined_events = combine_all_events(save_dir_events)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {total_processed}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_processed - total_successful}")
    print(f"Data saved to: {save_dir_data}")
    print(f"Processing plots saved to: {save_dir_plots}")
    if EVENT_DETECTION_OPTIONS['enable_event_detection']:
        print(f"Event data saved to: {save_dir_events}")
        print(f"Event plots saved to: {save_dir_event_plots}")
    
    print("\n=== FINAL WORKFLOW CONFIRMATION ===")
    print("✓ Early normalization ENABLED (applied before high-pass filter)")
    print("✓ All data on consistent [-1,1] scale throughout pipeline")
    print("✓ Event detection runs on normalized data with 2.5σ threshold")
    print("✓ Plots show both grey (normalized+high-pass) and black (final processed) on same scale")
    
if __name__ == "__main__":
    main()