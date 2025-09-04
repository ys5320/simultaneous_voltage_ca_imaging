import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from pathlib import Path
import numpy as np
from scipy import stats
from collections import Counter
from scipy.signal import butter, filtfilt, detrend
from scipy.signal import detrend
from scipy.signal import medfilt
import os
import re
import sys

def apply_linear_detrend_after_centering(data_array):
    """Apply linear detrending to already mean-centered data"""
    detrended_data = np.zeros_like(data_array)
    
    for i in range(data_array.shape[0]):
        # Apply linear detrend to mean-centered data
        detrended_data[i, :] = detrend(data_array[i, :], type='linear')
    
    return detrended_data

def apply_mean_centering(data_array, offset_value=0):
    """
    Mean-center each channel and optionally add offset
    """
    centered_data = np.zeros_like(data_array)
    
    for i in range(data_array.shape[0]):  # For each channel/ROI
        channel_data = data_array[i, :]
        channel_mean = np.mean(channel_data)
        
        # Subtract mean from each timepoint
        centered_data[i, :] = channel_data - channel_mean + offset_value
    
    return centered_data

def apply_median_baseline_correction(data, sampling_rate_hz=5, window_sec=120):
    """
    Use median filter to estimate and remove baseline drift
    Preserves spikes and oscillations better than high-pass filters
    """
    # Convert to samples (120 seconds = 600 samples at 5Hz)
    window_samples = int(window_sec * sampling_rate_hz)
    if window_samples % 2 == 0:
        window_samples += 1  # Median filter needs odd window size
    
    # Estimate baseline using median filter
    baseline = medfilt(data.astype(float), kernel_size=window_samples)
    
    # Remove baseline drift
    corrected_data = data - baseline
    
    return corrected_data

def apply_highpass_filter(data, sampling_rate_hz=5, cutoff_freq=0.01):
    """
    Apply high-pass filter to remove slow drift
    
    Parameters:
    -----------
    data : np.array
        Channel data to filter
    sampling_rate_hz : float
        Sampling rate in Hz
    cutoff_freq : float
        High-pass cutoff frequency in Hz (0.01 Hz = removes trends slower than 100 seconds)
    
    Returns:
    --------
    filtered_data : np.array
        High-pass filtered data
    """
    # Design Butterworth high-pass filter
    nyquist = sampling_rate_hz / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    
    # Apply zero-phase filter (forward and backward to avoid phase shift)
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def ensure_numeric_data(data):
    """Convert data to numeric format, handling pandas DataFrames and various data types"""
    if hasattr(data, 'values'):  # pandas DataFrame/Series
        data_array = data.values
    else:
        data_array = np.array(data)
    
    # Convert to float64 to ensure numeric operations work
    try:
        # Handle object dtype or mixed types
        if data_array.dtype == 'object' or not np.issubdtype(data_array.dtype, np.number):
            import pandas as pd
            data_array = pd.to_numeric(data_array.flatten(), errors='coerce').reshape(data_array.shape)
        
        # Convert to float64 for consistent numeric operations
        data_array = data_array.astype(np.float64)
        
        # Check for any NaN values and handle them
        if np.any(np.isnan(data_array)):
            print(f"Warning: Found {np.sum(np.isnan(data_array))} NaN values in data. Replacing with column means.")
            # Replace NaN with column means
            for i in range(data_array.shape[0]):
                channel_data = data_array[i, :]
                if np.any(np.isnan(channel_data)):
                    channel_mean = np.nanmean(channel_data)
                    channel_data[np.isnan(channel_data)] = channel_mean
                    data_array[i, :] = channel_data
        
        return data_array
        
    except Exception as e:
        print(f"Error converting data to numeric: {e}")
        raise

def detect_sudden_rise_gradient(data, sampling_rate_hz=5, time_window=(15, 20), threshold_percentile=95, min_duration=5, single_detection=True):
    """
    Method 1: Gradient-based detection with time window constraint
    Detects sudden rises by finding points where the gradient exceeds a threshold
    Only searches within specified time window (in minutes)
    """
    samples_per_min = sampling_rate_hz * 60
    # Convert time window to sample indices
    if isinstance(time_window[0], int) and time_window[0] > 100:
        # Assume frame indices if values are large integers
        start_idx = time_window[0]
        end_idx = time_window[1]
    else:
        # Assume minutes if values are small floats
        samples_per_min = sampling_rate_hz * 60
        start_idx = int(time_window[0] * samples_per_min)
        end_idx = int(time_window[1] * samples_per_min)
    
    # Ensure indices are within data bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(data), end_idx)
    
    # Extract window data
    window_data = data[start_idx:end_idx]
    
    # Calculate gradient (first derivative) only for window
    gradient_window = np.gradient(window_data)
    
    # Set threshold based on percentile of positive gradients in window
    positive_gradients = gradient_window[gradient_window > 0]
    if len(positive_gradients) > 0:
        threshold = np.percentile(positive_gradients, threshold_percentile)
    else:
        threshold = np.std(gradient_window) * 2
    
    # Find points above threshold
    rise_points = gradient_window > threshold
    
    # Find start of sustained rises
    rise_starts_relative = []
    in_rise = False
    rise_start = 0
    
    for i, is_rising in enumerate(rise_points):
        if is_rising and not in_rise:
            rise_start = i
            in_rise = True
        elif not is_rising and in_rise:
            if i - rise_start >= min_duration:
                rise_starts_relative.append(rise_start)
            in_rise = False
    
    # Handle case where rise continues to end of window
    if in_rise and len(rise_points) - rise_start >= min_duration:
        rise_starts_relative.append(rise_start)
    
    # If single_detection is True, find only the most significant rise
    if single_detection and len(rise_starts_relative) > 0:
        # Find the rise with the highest peak gradient
        max_gradient = -np.inf
        best_rise_idx = 0
        
        for rise_idx in rise_starts_relative:
            # Look at next 30 samples to find peak gradient
            end_search = min(rise_idx + int(0.1 * sampling_rate_hz * 60), len(gradient_window))
            peak_grad = np.max(gradient_window[rise_idx:end_search])
            
            if peak_grad > max_gradient:
                max_gradient = peak_grad
                best_rise_idx = rise_idx
        
        rise_starts_relative = [best_rise_idx]
    
    # Convert relative indices back to absolute indices
    rise_starts_absolute = [idx + start_idx for idx in rise_starts_relative]
    
    return rise_starts_absolute, gradient_window, threshold, (start_idx, end_idx)

def detect_sudden_rise_zscore(data, sampling_rate_hz=5, time_window=(15, 20), window_size=100, threshold=3, min_rise=None, single_detection=True):
    """
    Method 2: Z-score based detection with time window constraint
    Detects anomalous rises compared to local baseline within specified time range
    """
    samples_per_min = sampling_rate_hz * 60
    # Convert time window to sample indices
    if isinstance(time_window[0], int) and time_window[0] > 100:
        # Assume frame indices if values are large integers
        start_idx = time_window[0]
        end_idx = time_window[1]
    else:
        # Assume minutes if values are small floats
        samples_per_min = sampling_rate_hz * 60
        start_idx = int(time_window[0] * samples_per_min)
        end_idx = int(time_window[1] * samples_per_min)
    
    # Ensure indices are within data bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(data), end_idx)
    
    # Use broader context for baseline calculation (5 minutes before window)
    baseline_start = max(0, start_idx - int(5 * samples_per_min))
    baseline_data = data[baseline_start:start_idx]
    window_data = data[start_idx:end_idx]
    
    # Calculate baseline statistics from pre-window data
    if len(baseline_data) > window_size:
        baseline_mean = np.mean(baseline_data[-window_size:])
        baseline_std = np.std(baseline_data[-window_size:])
    else:
        baseline_mean = np.mean(data[:start_idx]) if start_idx > 0 else np.mean(data)
        baseline_std = np.std(data[:start_idx]) if start_idx > 0 else np.std(data)
    
    # Calculate z-score for window data
    z_scores = (window_data - baseline_mean) / (baseline_std + 1e-8)
    
    # Find points with high positive z-scores
    rise_points = z_scores > threshold
    
    # Optional: also check for minimum absolute rise
    if min_rise is not None:
        abs_rise = window_data - baseline_mean
        rise_points = rise_points & (abs_rise > min_rise)
    
    # Find rise start points (relative to window)
    rise_starts_relative = np.where(np.diff(rise_points.astype(int)) == 1)[0] + 1
    
    # Add first point if it starts with a rise
    if len(rise_points) > 0 and rise_points[0]:
        rise_starts_relative = np.concatenate([[0], rise_starts_relative])
    
    # If single_detection is True, find only the most significant rise
    if single_detection and len(rise_starts_relative) > 0:
        # Find the rise with the highest z-score
        max_z_score = -np.inf
        best_rise_idx = 0
        
        for rise_idx in rise_starts_relative:
            # Look at next 30 samples (6 seconds) to find peak z-score
            end_search = min(rise_idx + int(0.1 * sampling_rate_hz * 60), len(z_scores))
            peak_z = np.max(z_scores[rise_idx:end_search])
            
            if peak_z > max_z_score:
                max_z_score = peak_z
                best_rise_idx = rise_idx
        
        rise_starts_relative = [best_rise_idx]
    
    # Convert to absolute indices
    rise_starts_absolute = [idx + start_idx for idx in rise_starts_relative]
    
    return rise_starts_absolute, z_scores, baseline_mean, (start_idx, end_idx)

def detect_sudden_rise_cusum(data, sampling_rate_hz=5, time_window=(15, 20), threshold=None, drift=None):
    """
    Method 3: CUSUM change point detection with time window constraint
    Good for detecting sustained changes in signal level within specified time range
    """
    samples_per_min = sampling_rate_hz * 60
    # Convert time window to sample indices
    if isinstance(time_window[0], int) and time_window[0] > 100:
        # Assume frame indices if values are large integers
        start_idx = time_window[0]
        end_idx = time_window[1]
    else:
        # Assume minutes if values are small floats
        samples_per_min = sampling_rate_hz * 60
        start_idx = int(time_window[0] * samples_per_min)
        end_idx = int(time_window[1] * samples_per_min)
    
    # Ensure indices are within data bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(data), end_idx)
    
    # Extract window data
    window_data = data[start_idx:end_idx]
    
    # Set thresholds based on window data statistics if not provided
    if threshold is None:
        threshold = 2 * np.std(window_data)
    if drift is None:
        drift = np.std(window_data) / 2
    
    # Calculate differences
    diff = np.diff(window_data)
    
    # CUSUM algorithm
    cusum_pos = np.zeros(len(diff))
    
    for i in range(1, len(diff)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + diff[i-1] - drift)
    
    # Detect change points
    rise_points = cusum_pos > threshold
    rise_starts_relative = np.where(np.diff(rise_points.astype(int)) == 1)[0] + 1
    
    # Add first point if it starts with a rise
    if len(rise_points) > 0 and rise_points[0]:
        rise_starts_relative = np.concatenate([[0], rise_starts_relative])
    
    # Convert to absolute indices
    rise_starts_absolute = [idx + start_idx for idx in rise_starts_relative]
    
    return rise_starts_absolute, cusum_pos, threshold, (start_idx, end_idx)

def detect_sudden_rise_edge_detection(data, sampling_rate_hz=5, time_window=(15, 20), sigma=2, threshold_percentile=90):
    """
    Method 4: Edge detection approach with time window constraint
    Uses Gaussian smoothing and edge detection to find sharp rises
    """
    samples_per_min = sampling_rate_hz * 60
    # Convert time window to sample indices
    if isinstance(time_window[0], int) and time_window[0] > 100:
        # Assume frame indices if values are large integers
        start_idx = time_window[0]
        end_idx = time_window[1]
    else:
        # Assume minutes if values are small floats
        samples_per_min = sampling_rate_hz * 60
        start_idx = int(time_window[0] * samples_per_min)
        end_idx = int(time_window[1] * samples_per_min)
    
    # Ensure indices are within data bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(data), end_idx)
    
    # Extract window data
    window_data = data[start_idx:end_idx]
    
    # Smooth the data
    smoothed = gaussian_filter1d(window_data, sigma=sigma)
    
    # Calculate second derivative (edge detection)
    second_deriv = np.gradient(np.gradient(smoothed))
    
    # Find significant negative second derivatives (start of rise)
    negative_second_deriv = -second_deriv[second_deriv < 0]
    if len(negative_second_deriv) > 0:
        threshold = np.percentile(negative_second_deriv, threshold_percentile)
        edge_points = -second_deriv > threshold
    else:
        edge_points = np.zeros(len(second_deriv), dtype=bool)
    
    # Find edge starts (relative to window)
    rise_starts_relative = np.where(np.diff(edge_points.astype(int)) == 1)[0] + 1
    
    # Add first point if it starts with an edge
    if len(edge_points) > 0 and edge_points[0]:
        rise_starts_relative = np.concatenate([[0], rise_starts_relative])
    
    # Convert to absolute indices
    rise_starts_absolute = [idx + start_idx for idx in rise_starts_relative]
    
    return rise_starts_absolute, second_deriv, smoothed, (start_idx, end_idx)

def detect_multiple_channels(data_matrix, sampling_rate_hz=5, time_window=(15, 20), method='zscore', consensus_threshold=0.5, apply_filter=True, **kwargs):
    """
    Apply rise detection to multiple channels within time window and find consensus
    data_matrix: shape (n_channels, n_timepoints)
    consensus_threshold: fraction of channels that must detect a rise for consensus
    """
    # Ensure data is numeric
    data_array = ensure_numeric_data(data_matrix)
    
    # Apply high-pass filter to remove drift (add this section)
    if apply_filter:
        print("Applying high-pass filter, mean centering, and linear detrending...")
        filtered_data = np.zeros_like(data_array)
        for i in range(data_array.shape[0]):
            # Step 1: Apply high-pass filter (removes slow drift)
            high_pass_filtered = apply_highpass_filter(data_array[i, :], sampling_rate_hz, cutoff_freq=0.01)
            # Step 2: Mean center the data
            channel_mean = np.mean(high_pass_filtered)
            mean_centered = high_pass_filtered - channel_mean
            # Step 3: Remove linear trends
            filtered_data[i, :] = detrend(mean_centered, type='linear')
        data_array = filtered_data
    '''
    # Apply median filter baseline removal
    if apply_filter:
        print("Applying median baseline correction, mean centering, and linear detrending...")
        filtered_data = np.zeros_like(data_array)
        for i in range(data_array.shape[0]):
            # Step 1: Remove baseline drift
            baseline_corrected = apply_median_baseline_correction(data_array[i, :], sampling_rate_hz)
            # Step 2: Mean center the data
            channel_mean = np.mean(baseline_corrected)
            mean_centered = baseline_corrected - channel_mean
            # Step 3: Remove linear trends
            filtered_data[i, :] = detrend(mean_centered, type='linear')
        data_array = filtered_data
    '''
    all_detections = []
    channel_results = {}
    
    for i, channel_data in enumerate(data_array):
        try:
            if method == 'gradient':
                detections, *other_data = detect_sudden_rise_gradient(
                    channel_data, sampling_rate_hz, time_window, single_detection=kwargs.get('single_detection', True), **{k:v for k,v in kwargs.items() if k != 'single_detection'})
            elif method == 'zscore':
                detections, *other_data = detect_sudden_rise_zscore(
                    channel_data, sampling_rate_hz, time_window, single_detection=kwargs.get('single_detection', True), **{k:v for k,v in kwargs.items() if k != 'single_detection'})
            elif method == 'cusum':
                detections, *other_data = detect_sudden_rise_cusum(
                    channel_data, sampling_rate_hz, time_window, **kwargs)
            elif method == 'edge':
                detections, *other_data = detect_sudden_rise_edge_detection(
                    channel_data, sampling_rate_hz, time_window, **kwargs)
            
            channel_results[f'channel_{i}'] = {
                'detections': detections,
                'other_data': other_data
            }
            
            all_detections.extend([(det, i) for det in detections])
            
        except Exception as e:
            print(f"Warning: Error processing channel {i}: {e}")
            channel_results[f'channel_{i}'] = {
                'detections': [],
                'other_data': []
            }
    
    # Find consensus detections (within ±30 samples of each other)
    if all_detections:
        # Convert time window to sample indices for reference
        samples_per_min = sampling_rate_hz * 60
        start_idx = int(time_window[0] * samples_per_min)
        end_idx = int(time_window[1] * samples_per_min)
        window_size = end_idx - start_idx
        
        # Group nearby detections
        consensus_points = []
        detection_times = [det[0] for det in all_detections]
        detection_times.sort()
        
        tolerance = int(0.1 * sampling_rate_hz * 60)  # ±6 seconds tolerance at 5Hz
        
        i = 0
        while i < len(detection_times):
            current_time = detection_times[i]
            group = [current_time]
            
            # Find all detections within tolerance
            j = i + 1
            while j < len(detection_times) and detection_times[j] - current_time <= tolerance:
                group.append(detection_times[j])
                j += 1
            
            # Check if enough channels detected this rise
            if len(group) >= consensus_threshold * len(data_array):
                consensus_points.append(np.mean(group))
            
            i = j if j > i + 1 else i + 1
        
        return consensus_points, channel_results, (start_idx, end_idx)
    
    return [], channel_results, None

def convert_time_to_samples(time_minutes, sampling_rate_hz=5):
    """Helper function to convert time in minutes to sample index"""
    return int(time_minutes * sampling_rate_hz * 60)

def convert_samples_to_time(sample_index, sampling_rate_hz=5):
    """Helper function to convert sample index to time in minutes"""
    return sample_index / (sampling_rate_hz * 60)

def run_detection_and_plot(data_matrix, sampling_rate_hz=5, time_window=(15, 20), 
                          save_dir=None):
    """Complete workflow: detect rises and create visualization"""
    
    # Ensure data is numeric and properly formatted
    print("Converting data to numeric format...")
    data_array = ensure_numeric_data(data_matrix)
    
    print(f"Data shape: {data_array.shape}")
    if isinstance(time_window[0], int) and time_window[0] > 100:
        print(f"Detection window: {time_window[0]}-{time_window[1]} frames")
    else:
        print(f"Detection window: {time_window[0]}-{time_window[1]} minutes")
    
    # Run multi-channel detection with single detection per channel
    consensus_points, channel_results, window_bounds = detect_multiple_channels(
        data_array,
        sampling_rate_hz=sampling_rate_hz,
        time_window=time_window,
        method='zscore',
        consensus_threshold=0.5,
        threshold=2.5,
        single_detection=True
    )
    
    # Convert results to minutes for reporting
    samples_per_min = sampling_rate_hz * 60
    
    print(f"\nDetection Results:")
    total_detections = 0
    channels_with_detection = 0
    
    for channel_name, results in channel_results.items():
        detections = results['detections']
        if detections:
            times = [f'{det/samples_per_min:.2f} min' for det in detections]
            print(f"{channel_name}: {times[0]}")
            total_detections += len(detections)
            channels_with_detection += 1
    
    print(f"Summary: {channels_with_detection}/{len(channel_results)} channels detected rises")
    
    # Create the plot
    n_channels, n_timepoints = data_array.shape
    time_minutes = np.arange(n_timepoints) / (sampling_rate_hz * 60)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate appropriate channel spacing based on data range
    data_ranges = [np.max(data_array[i, :]) - np.min(data_array[i, :]) for i in range(n_channels)]
    max_range = max(data_ranges) if data_ranges else 100
    channel_spacing = max_range * 1.5  # Space channels by 1.5x their data range
    
    print(f"Using channel spacing: {channel_spacing:.2f}")
    
    # Plot each channel with vertical offset
    for i in range(n_channels):
        channel_data = data_array[i, :]
        offset = i * channel_spacing
        
        # Plot the channel data with more visible line
        ax.plot(time_minutes, channel_data + offset, 'b-', linewidth=1.0, alpha=0.8)
        
        # Add channel label on the left
        ax.text(-1, np.mean(channel_data) + offset, f'{i}', 
                verticalalignment='center', fontsize=9, fontweight='bold')
        
        # Plot detection point for this channel
        channel_key = f'channel_{i}'
        if channel_key in channel_results and channel_results[channel_key]['detections']:
            detection_idx = channel_results[channel_key]['detections'][0]
            detection_time = detection_idx / samples_per_min
            
            # Get the y-value at the detection point
            y_val = channel_data[detection_idx] + offset
            
            # Plot vertical line spanning the channel's data range
            channel_min = np.min(channel_data) + offset
            channel_max = np.max(channel_data) + offset
            
            ax.plot([detection_time, detection_time], [channel_min, channel_max], 
                   color='red', linestyle='-', linewidth=2, alpha=0.8)
            
            # Add detection marker
            ax.plot(detection_time, y_val, 'ro', markersize=5, alpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title('Multi-Channel Voltage Data with Rise Detection', fontsize=14, fontweight='bold')
    
    # Set appropriate axis limits
    ax.set_xlim(0, np.max(time_minutes))
    ax.set_ylim(-channel_spacing*0.5, (n_channels-0.5) * channel_spacing)
    ax.set_yticks([])
    
    # Add grid for time axis only
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1, label='Voltage Data'),
        Line2D([0], [0], color='red', linewidth=2, label='Rise Detection')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    if save_dir is not None:
        save_path = Path(save_dir) / "voltage_rise_detection.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return consensus_points, channel_results

def find_consensus_timepoint(channel_results, sampling_rate_hz=5, method='median', tolerance_seconds=30):
    """
    Find the consensus timepoint from all channel detections and assign to all channels
    
    Parameters:
    -----------
    channel_results : dict
        Results from detect_multiple_channels
    sampling_rate_hz : float
        Sampling rate in Hz
    method : str
        'median', 'mode', 'mean', or 'density_peak'
    tolerance_seconds : float
        Tolerance for grouping detections (in seconds)
    
    Returns:
    --------
    consensus_timepoint : int
        Sample index of consensus timepoint
    all_channel_results : dict
        Updated results with consensus timepoint assigned to all channels
    """
    
    # Collect all detection timepoints
    all_detections = []
    for channel_name, results in channel_results.items():
        detections = results['detections']
        all_detections.extend(detections)
    
    if not all_detections:
        print("No detections found across any channels!")
        return None, channel_results
    
    all_detections = np.array(all_detections)
    samples_per_min = sampling_rate_hz * 60
    tolerance_samples = int(tolerance_seconds * sampling_rate_hz)
    
    print(f"Found {len(all_detections)} total detections across channels")
    print(f"Detection times: {[f'{det/samples_per_min:.2f} min' for det in all_detections]}")
    
    # Method 1: Median (most robust to outliers)
    if method == 'median':
        consensus_timepoint = int(np.median(all_detections))
        print(f"Consensus method: Median = {consensus_timepoint/samples_per_min:.2f} min")
    
    # Method 2: Mode (most frequent timepoint within tolerance)
    elif method == 'mode':
        # Group detections within tolerance
        groups = []
        used = np.zeros(len(all_detections), dtype=bool)
        
        for i, detection in enumerate(all_detections):
            if used[i]:
                continue
            
            # Find all detections within tolerance
            group = [detection]
            used[i] = True
            
            for j in range(i+1, len(all_detections)):
                if not used[j] and abs(all_detections[j] - detection) <= tolerance_samples:
                    group.append(all_detections[j])
                    used[j] = True
            
            groups.append(group)
        
        # Find largest group (mode)
        largest_group = max(groups, key=len)
        consensus_timepoint = int(np.mean(largest_group))
        print(f"Consensus method: Mode group size = {len(largest_group)}, time = {consensus_timepoint/samples_per_min:.2f} min")
    
    # Method 3: Mean
    elif method == 'mean':
        consensus_timepoint = int(np.mean(all_detections))
        print(f"Consensus method: Mean = {consensus_timepoint/samples_per_min:.2f} min")
    
    # Method 4: Density peak (find peak of kernel density estimate)
    elif method == 'density_peak':
        from scipy.stats import gaussian_kde
        
        # Create density estimate
        kde = gaussian_kde(all_detections, bw_method=0.1)
        
        # Evaluate density across the range
        time_range = np.arange(np.min(all_detections) - tolerance_samples, 
                              np.max(all_detections) + tolerance_samples, 
                              sampling_rate_hz)  # 1-second resolution
        densities = kde(time_range)
        
        # Find peak
        peak_idx = np.argmax(densities)
        consensus_timepoint = int(time_range[peak_idx])
        print(f"Consensus method: Density peak = {consensus_timepoint/samples_per_min:.2f} min")
    
    # Create new results with consensus timepoint assigned to ALL channels
    all_channel_results = {}
    n_channels = len(channel_results)
    
    for channel_name, results in channel_results.items():
        all_channel_results[channel_name] = {
            'detections': [consensus_timepoint],  # Assign consensus to all channels
            'other_data': results['other_data'],
            'original_detections': results['detections'],  # Keep original for reference
            'method': 'consensus'
        }
    
    print(f"\nConsensus timepoint: {consensus_timepoint/samples_per_min:.2f} min (sample {consensus_timepoint})")
    print(f"Assigned consensus timepoint to all {n_channels} channels")
    
    return consensus_timepoint, all_channel_results

def analyze_detection_consistency(channel_results, sampling_rate_hz=5):
    """
    Analyze the consistency of detections across channels
    """
    samples_per_min = sampling_rate_hz * 60
    
    # Get all detection times
    detection_times = []
    channel_names = []
    
    for channel_name, results in channel_results.items():
        detections = results['detections']
        if detections:
            detection_times.extend(detections)
            channel_names.extend([channel_name] * len(detections))
    
    if not detection_times:
        print("No detections found!")
        return
    
    detection_times = np.array(detection_times)
    detection_minutes = detection_times / samples_per_min
    
    print("\nDetection Consistency Analysis:")
    print(f"Total detections: {len(detection_times)}")
    print(f"Time range: {np.min(detection_minutes):.2f} - {np.max(detection_minutes):.2f} min")
    print(f"Time span: {np.max(detection_minutes) - np.min(detection_minutes):.2f} min")
    print(f"Standard deviation: {np.std(detection_minutes):.3f} min")
    print(f"Median: {np.median(detection_minutes):.2f} min")
    
    # Show histogram of detection times
    print(f"\nDetection time distribution (30-second bins):")
    bin_size = 0.5  # 30 seconds
    bins = np.arange(np.min(detection_minutes), np.max(detection_minutes) + bin_size, bin_size)
    hist, bin_edges = np.histogram(detection_minutes, bins=bins)
    
    for i, count in enumerate(hist):
        if count > 0:
            print(f"  {bin_edges[i]:.1f}-{bin_edges[i+1]:.1f} min: {count} detections")

def run_detection_with_consensus(data_matrix, trial_string, toxin, detection_method = 'zscore', sampling_rate_hz=5, time_window=(15, 20), 
                                save_dir=None, consensus_method='median', apply_filter = True):
    """
    Complete workflow with consensus timepoint assignment
    """
    
    # Original detection
    print("=== STEP 1: Individual Channel Detection ===")
    consensus_points, channel_results, window_bounds = detect_multiple_channels(
        ensure_numeric_data(data_matrix),
        sampling_rate_hz=sampling_rate_hz,
        time_window=time_window,
        method= detection_method, # gradient, zscore, cumsum, edge
        consensus_threshold=0.5,
        threshold=2.5,
        single_detection=True,
        apply_filter=apply_filter
    )
    
    # Analyze consistency
    print("\n=== STEP 2: Detection Consistency Analysis ===")
    analyze_detection_consistency(channel_results, sampling_rate_hz)
    
    # Find consensus timepoint
    print(f"\n=== STEP 3: Consensus Timepoint ({consensus_method.upper()}) ===")
    consensus_timepoint, final_results = find_consensus_timepoint(
        channel_results, sampling_rate_hz, method=consensus_method)
    
    if consensus_timepoint is None:
        return None, channel_results
    
    # Create plot with consensus timepoint for all channels
    print("\n=== STEP 4: Creating Plot with Consensus ===")
    data_array = ensure_numeric_data(data_matrix)
    n_channels, n_timepoints = data_array.shape
    time_minutes = np.arange(n_timepoints) / (sampling_rate_hz * 60)
    
    fig, ax = plt.subplots(figsize=(10,20))
    
    # Calculate channel spacing
    data_ranges = [np.max(data_array[i, :]) - np.min(data_array[i, :]) for i in range(n_channels)]
    max_range = max(data_ranges) if data_ranges else 100
    channel_spacing = max_range * 0.5
    
    samples_per_min = sampling_rate_hz * 60
    consensus_time_min = consensus_timepoint / samples_per_min
    
    #n_channels = 10

    # With this:
    exclude_frame = 200
    # Calculate timepoints for consensus, ±200, +400, and absolute frame 200
    consensus_minus_200 = max(0, consensus_timepoint - 200)
    consensus_plus_200 = min(n_timepoints - 1, consensus_timepoint + 200)
    consensus_plus_400 = min(n_timepoints - 1, consensus_timepoint + 400)
    absolute_frame_200 = 200

    consensus_minus_200_time = consensus_minus_200 / samples_per_min
    consensus_plus_200_time = consensus_plus_200 / samples_per_min
    consensus_plus_400_time = consensus_plus_400 / samples_per_min
    absolute_frame_200_time = absolute_frame_200 / samples_per_min
        
    print(f"Plotting lines at:")
    print(f"  Absolute frame 200: {absolute_frame_200_time:.2f} min")
    print(f"  Consensus - 200: {consensus_minus_200_time:.2f} min")
    print(f"  Consensus: {consensus_time_min:.2f} min") 
    print(f"  Consensus + 200: {consensus_plus_200_time:.2f} min")
    print(f"  Consensus + 400: {consensus_plus_400_time:.2f} min")
    
    # Plot each channel
    for i in range(n_channels):
        channel_data = data_array[i, :]
        offset = i * channel_spacing
        
        # Plot channel data
        ax.plot(time_minutes, channel_data + offset, 'b-', linewidth=1.0, alpha=0.7)
        
        # Channel label
        ax.text(-1, np.mean(channel_data) + offset, f'{i}', 
                verticalalignment='center', fontsize=9, fontweight='bold')
        
        # Calculate y-range for this channel
        channel_min = np.min(channel_data) + offset
        channel_max = np.max(channel_data) + offset
        
        # Plot vertical line at absolute frame 200 (purple, dashed)
        ax.plot([absolute_frame_200_time, absolute_frame_200_time], [channel_min, channel_max], 
            color='purple', linestyle='--', linewidth=1.5, alpha=0.7)

        # Plot vertical line at consensus - 200 (green, dashed)
        ax.plot([consensus_minus_200_time, consensus_minus_200_time], [channel_min, channel_max], 
            color='green', linestyle='--', linewidth=1.5, alpha=0.7)

        # Plot vertical line at consensus time (red, solid)
        ax.plot([consensus_time_min, consensus_time_min], [channel_min, channel_max], 
            color='red', linestyle='-', linewidth=2, alpha=0.8)

        # Plot vertical line at consensus + 200 (orange, dashed)
        ax.plot([consensus_plus_200_time, consensus_plus_200_time], [channel_min, channel_max], 
            color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        # Plot vertical line at consensus + 400 (blue, dashed)
        ax.plot([consensus_plus_400_time, consensus_plus_400_time], [channel_min, channel_max], 
            color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Detection markers at all three timepoints
        # Detection markers at all five timepoints
        y_val_frame_200 = channel_data[absolute_frame_200] + offset
        y_val_minus_200 = channel_data[consensus_minus_200] + offset
        y_val_consensus = channel_data[consensus_timepoint] + offset
        y_val_plus_200 = channel_data[consensus_plus_200] + offset
        y_val_plus_400 = channel_data[consensus_plus_400] + offset

        ax.plot(absolute_frame_200_time, y_val_frame_200, 'o', color='purple', markersize=4, alpha=0.8)
        ax.plot(consensus_minus_200_time, y_val_minus_200, 'go', markersize=4, alpha=0.8)
        ax.plot(consensus_time_min, y_val_consensus, 'ro', markersize=5, alpha=0.9)
        ax.plot(consensus_plus_200_time, y_val_plus_200, 'o', color='orange', markersize=4, alpha=0.8)
        ax.plot(consensus_plus_400_time, y_val_plus_400, 'o', color='blue', markersize=4, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(f'Multi-Channel Voltage Data with Consensus Detection ({consensus_method.upper()})', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, np.max(time_minutes))
    ax.set_ylim(-channel_spacing*0.5, (n_channels-0.5) * channel_spacing)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1, label='Voltage Data'),
        Line2D([0], [0], color='purple', linewidth=1.5, linestyle='--', label=f'Frame 200 ({absolute_frame_200_time:.2f} min)'),
        Line2D([0], [0], color='green', linewidth=1.5, linestyle='--', label=f'Consensus - 200 ({consensus_minus_200_time:.2f} min)'),
        Line2D([0], [0], color='red', linewidth=2, label=f'Consensus Detection ({consensus_time_min:.2f} min)'),
        Line2D([0], [0], color='orange', linewidth=1.5, linestyle='--', label=f'Consensus + 200 ({consensus_plus_200_time:.2f} min)'),
        Line2D([0], [0], color='blue', linewidth=1.5, linestyle='--', label=f'Consensus + 400 ({consensus_plus_400_time:.2f} min)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    if save_dir is not None:
        try:
            from pathlib import Path
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            full_path = save_path / f"{toxin}_{trial_string}_voltage_consensus_detection_{detection_method}_{consensus_method}.png"
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {full_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    
    #plt.show()
    
    return consensus_timepoint, final_results

if __name__ == "__main__":
    
    homd = Path.home()
    if "ys5320" in str(home):
        top_dir = Path(home, "firefly_link")
    else:
        top_dir = Path(r'R:\home\firefly_link')
    
    data_dir = Path(top_dir, r'ca_voltage_imaging_working\results')
    save_dir = Path(top_dir, r'Calcium_Voltage_Imaging\code_yilin\infusion_detection_check\highpass_0.01Hz_detrend')
    toxins = ['L-15_control','Ani9_10uM','ATP_1mM','TRAM-34_1uM','dantrolene_10uM']
    for toxin in toxins:
        # Find all CSV files with case-insensitive matching
        all_csv_files = list(data_dir.glob('*_voltage_transfected*.csv'))
        csv_files = []
        
        for file in all_csv_files:
            # Handle special case for dantrolene (case-insensitive)
            if toxin.lower() == 'dantrolene_10um':
                if 'dantrolene_10um' in file.name.lower():
                    csv_files.append(file)
            else:
                if toxin.lower() in file.name.lower():
                    csv_files.append(file)
        pattern = f'*{toxin}_voltage_transfected*.csv (case-insensitive)'
        
        print(f"Found {len(csv_files)} files matching pattern '{pattern}':")
        for file in csv_files:
            print(f"  - {file.name}")
        
        if not csv_files:
            print("No files found! Check the data directory and pattern.")
        else:
            print(f"\nProcessing {len(csv_files)} files...")
            print("=" * 60)
            
            # Process each file
            for i, csv_file in enumerate(csv_files, 1):
                print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file.name}")
                print("-" * 50)
                
                try:
                    # Extract trial name (remove .csv extension)
                    trial = csv_file.stem
                    trial_string = '_'.join(trial.split('_')[:3])
                    
                    print(f"Trial: {trial}")
                    print(f"Trial string: {trial_string}")
                    
                    # Load data
                    data_matrix = pd.read_csv(csv_file)
                    data_matrix = data_matrix.iloc[:, -10000:]
                    
                    # Run consensus detection
                    consensus_timepoint, final_results = run_detection_with_consensus(
                        data_matrix, 
                        trial_string=trial_string,
                        toxin = toxin,
                        save_dir=save_dir,
                        detection_method = 'zscore',
                        time_window=(4500, 5500),
                        consensus_method='mode',  # 'median','mode','density_peak', 'mean'
                        apply_filter = True
                    )
                    
                    if consensus_timepoint is not None:
                        samples_per_min = 5 * 60
                        print(f"SUCCESS: Consensus at {consensus_timepoint/samples_per_min:.2f} min")
                    else:
                        print("WARNING: No consensus found")
                    
                except Exception as e:
                    print(f"ERROR processing {csv_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                print("=" * 60)
            
            print(f"\nCompleted processing {len(csv_files)} files.")
            print(f"Results saved to: {save_dir}")

