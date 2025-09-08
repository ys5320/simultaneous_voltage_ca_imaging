import numpy as np
import pandas as pd
from pathlib import Path
import tifffile as tiff
import os
import functions as canf

# Import only the essential functions we actually need
from smart_detection_pipeline import (
    detect_file_type_and_datapoints,
    apply_highpass_to_full_timeseries,
    find_consensus_timepoint_from_voltage,
    segment_highpass_timeseries,
    apply_segment_processing,
    save_segment_data,
    detect_voltage_events_enhanced_threshold,
    plot_events_with_arrows_global_fixed,
    filter_simultaneous_events,
    filter_bleaching_artifacts,
    process_events_for_segments_with_enhanced_filter
)

def determine_toxin_from_trial(trial_string, filename):
    """Determine toxin type from trial string or filename"""
    filename_lower = filename.lower()
    trial_lower = trial_string.lower()
    
    if 'l-15' in filename_lower or 'l-15' in trial_lower:
        return 'L-15_control'
    elif 'ani9' in filename_lower or 'ani9' in trial_lower:
        return 'Ani9_10uM'
    elif 'atp' in filename_lower or 'atp' in trial_lower:
        return 'ATP_1mM'
    elif 'tram' in filename_lower or 'tram' in trial_lower:
        return 'TRAM-34_1uM'
    elif 'dantrolene' in filename_lower or 'dantrolene' in trial_lower:
        return 'dantrolene_10uM'
    elif 'dmso' in filename_lower or 'dmso' in trial_lower:
        return 'DMSO_0.1%_control'
    else:
        return 'unknown'

def run_pipeline(df_file, top_dir, data_dir, HPC_num=None):
    """
    HPC-compatible pipeline that mirrors make_videos structure
    """
    # Processing options
    PROCESSING_OPTIONS = {
        'apply_mean_center': False,
        'apply_detrend': False,
        'apply_gaussian': True,
        'gaussian_sigma': 3,
        'apply_normalization': True,
        'enable_event_detection': True,
        'threshold_multiplier': 2.5,
        'use_global_threshold': False,
        'max_simultaneous_cells': 999,
        'overlap_threshold': 0.9,
        'use_fixed_std': True,
        'fixed_std_value': 0.05,
        'filter_bleaching_artifacts': True,
        'bleaching_min_cells': 5,
        'bleaching_overlap_threshold': 0.7,
        'bleaching_start_time_threshold': 10.0,
    }
    
    # Set up directories - follow same pattern as make_videos
    base_results_dir = Path(top_dir, 'analysis', 'results_pipeline')
    save_dir_plots = Path(base_results_dir, 'timeseries_plots')
    save_dir_event_plots = Path(base_results_dir, 'event_detection_plots')
    
    # Create directories
    base_results_dir.mkdir(parents=True, exist_ok=True)
    save_dir_plots.mkdir(parents=True, exist_ok=True)
    save_dir_event_plots.mkdir(parents=True, exist_ok=True)
    
    # Load and filter dataframe - EXACTLY like make_videos
    df = df_file
    
    # Process trials - EXACTLY like make_videos loop structure
    for idx, trial_row in df.iterrows():
        if HPC_num is not None:  # HPC mode
            if idx != HPC_num:
                print(HPC_num, idx)
                continue
        
        trial_string = trial_row.trial_string
        original_folder = Path(trial_row.folder)
        
        print(f"\nProcessing pipeline for trial: {trial_string}")
        
        # Find corresponding timeseries files for this trial
        voltage_files = list(data_dir.glob(f"*{trial_string}*_voltage_transfected*.csv"))
        calcium_files = list(data_dir.glob(f"*{trial_string}*_ca_transfected*.csv"))
        
        if not voltage_files or not calcium_files:
            print(f"  ⚠ Missing timeseries files for {trial_string}, skipping")
            continue
        
        voltage_file = voltage_files[0]
        calcium_file = calcium_files[0]
        
        # Determine toxin
        toxin = determine_toxin_from_trial(trial_string, voltage_file.name)
        
        # Create trial-specific directory
        trial_data_dir = Path(base_results_dir, trial_string)
        trial_data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run the complete processing pipeline
            success = process_single_trial_complete(
                voltage_file, calcium_file, toxin, trial_string,
                original_folder, save_dir_plots, trial_data_dir, 
                trial_data_dir, save_dir_event_plots,
                **PROCESSING_OPTIONS
            )
            
            if success:
                print(f"✓ Pipeline completed for {trial_string}")
            else:
                print(f"✗ Pipeline failed for {trial_string}")
                
        except Exception as e:
            print(f"✗ Error in pipeline for {trial_string}: {e}")
            import traceback
            traceback.print_exc()
            continue

def process_single_trial_complete(voltage_file, ca_file, toxin, trial_string, 
                                original_folder, save_dir_plots, save_dir_data, 
                                save_dir_events, save_dir_event_plots,
                                apply_mean_center=True, apply_detrend=True, 
                                apply_gaussian=False, gaussian_sigma=3,
                                apply_normalization=True,
                                enable_event_detection=True, **event_options):
    """
    Complete processing pipeline for a single trial
    """
    print(f"Processing trial: {trial_string}")
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
        
        # Extract cell positions and IDs
        voltage_cell_positions = voltage_data[['cell_id', 'cell_x', 'cell_y']].copy()
        ca_cell_positions = ca_data[['cell_id', 'cell_x', 'cell_y']].copy()

        # Extract timeseries data (everything after cell_y)
        cell_y_idx_v = voltage_data.columns.get_loc('cell_y')
        cell_y_idx_c = ca_data.columns.get_loc('cell_y')
        voltage_timeseries = voltage_data.iloc[:, cell_y_idx_v + 1:]
        ca_timeseries = ca_data.iloc[:, cell_y_idx_c + 1:]

        print(f"Voltage cells: {len(voltage_cell_positions)}, Calcium cells: {len(ca_cell_positions)}")
        
        # Check if this is a post-only file
        has_post_in_name = 'post' in voltage_file.name.lower() or 'post' in ca_file.name.lower()
        
        if voltage_file_type == "post_only" or has_post_in_name:
            print("=== POST-ONLY FILES - SKIPPING SEGMENTATION ===")
            
            # Process as single post segment
            voltage_highpass = apply_highpass_to_full_timeseries(
                voltage_timeseries, sampling_rate_hz=5, cutoff_freq=0.01,
                apply_normalization=apply_normalization, data_type="voltage"
            )
            
            ca_highpass = apply_highpass_to_full_timeseries(
                ca_timeseries, sampling_rate_hz=5, cutoff_freq=0.01,
                apply_normalization=apply_normalization, data_type="calcium"
            )
            
            # Create single segment
            voltage_segments_raw = {'post': voltage_highpass}
            ca_segments_raw = {'post': ca_highpass}
            voltage_segments_processed = {}
            ca_segments_processed = {}
            
            # Process post segment for both modalities
            for data_type, segments_raw, cell_positions, filename in [
                ("voltage", voltage_segments_raw, voltage_cell_positions, voltage_file.name),
                ("calcium", ca_segments_raw, ca_cell_positions, ca_file.name)
            ]:
                processed, non_gaussian = apply_segment_processing(
                    segments_raw['post'],
                    apply_mean_center=apply_mean_center,
                    apply_detrend=apply_detrend,
                    apply_gaussian=apply_gaussian,
                    gaussian_sigma=gaussian_sigma,
                    data_type=data_type
                )
                
                if data_type == "voltage":
                    voltage_segments_processed['post'] = processed
                else:
                    ca_segments_processed['post'] = processed
                
                save_segment_data(processed, 'post', toxin, trial_string, 
                                save_dir_data, cell_positions=cell_positions, 
                                original_filename=filename, data_type=data_type)
                
                # ADD video creation for post only
                print("=== CREATING POST-ONLY VIDEO ===")
                try:
                    create_post_only_videos(
                        trial_string, original_folder, save_dir_data, sampling_rate_hz=5
                    )
                except Exception as e:
                    print(f"Warning: Could not create post-only videos for {trial_string}: {e}")
            
        else:
            print("=== FULL EXPERIMENT - USING SEGMENTATION ===")
            
            # Apply high-pass filtering
            voltage_highpass = apply_highpass_to_full_timeseries(
                voltage_timeseries, sampling_rate_hz=5, cutoff_freq=0.01,
                apply_normalization=apply_normalization, data_type="voltage"
            )
            
            ca_highpass = apply_highpass_to_full_timeseries(
                ca_timeseries, sampling_rate_hz=5, cutoff_freq=0.01,
                apply_normalization=apply_normalization, data_type="calcium"
            )
            
            # Find consensus timepoint using VOLTAGE data
            consensus_timepoint = find_consensus_timepoint_from_voltage(
                voltage_highpass, sampling_rate_hz=5, time_window=(4500, 5500),
                detection_method='zscore', consensus_method='mode'
            )
            
            if consensus_timepoint is None:
                print("✗ WARNING: No consensus found - cannot segment")
                return False
            
            # Create segment videos
            create_segment_videos(
                trial_string, consensus_timepoint, original_folder, 
                save_dir_data, sampling_rate_hz=5
            )
            
            # Segment both datasets using same consensus timepoint
            voltage_segments_raw = segment_highpass_timeseries(voltage_highpass, consensus_timepoint, data_type="voltage")
            ca_segments_raw = segment_highpass_timeseries(ca_highpass, consensus_timepoint, data_type="calcium")
            voltage_segments_processed = {}
            ca_segments_processed = {}
            
            # Process each segment for both modalities
            for segment_name in ['pre', 'post']:
                for data_type, segments_raw, cell_positions, filename in [
                    ("voltage", voltage_segments_raw, voltage_cell_positions, voltage_file.name),
                    ("calcium", ca_segments_raw, ca_cell_positions, ca_file.name)
                ]:
                    if segments_raw[segment_name] is not None:
                        print(f"\n=== PROCESSING {data_type.upper()} {segment_name.upper()} SEGMENT ===")
                        
                        processed, non_gaussian = apply_segment_processing(
                            segments_raw[segment_name],
                            apply_mean_center=apply_mean_center,
                            apply_detrend=apply_detrend,
                            apply_gaussian=apply_gaussian,
                            gaussian_sigma=gaussian_sigma,
                            data_type=data_type
                        )
                        
                        if data_type == "voltage":
                            voltage_segments_processed[segment_name] = processed
                        else:
                            ca_segments_processed[segment_name] = processed
                        
                        save_segment_data(processed, segment_name, toxin, trial_string, 
                                        save_dir_data, cell_positions=cell_positions, 
                                        original_filename=filename, data_type=data_type)
        
        # Event detection for both modalities
        if enable_event_detection:
            print(f"\n=== VOLTAGE EVENT DETECTION ===")
            voltage_events_df, voltage_excluded = process_events_for_segments_with_enhanced_filter(
                voltage_segments_raw, voltage_segments_processed, toxin, trial_string,
                save_dir_events, save_dir_event_plots,
                sampling_rate_hz=5, data_type="voltage", 
                cell_positions=voltage_cell_positions,
                has_post_in_name=has_post_in_name, **event_options
            )

            print(f"\n=== CALCIUM EVENT DETECTION ===")
            ca_events_df, ca_excluded = process_events_for_segments_with_enhanced_filter(
                ca_segments_raw, ca_segments_processed, toxin, trial_string,
                save_dir_events, save_dir_event_plots,
                sampling_rate_hz=5, data_type="calcium", 
                cell_positions=ca_cell_positions,
                has_post_in_name=has_post_in_name, **event_options
            )
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR processing {trial_string}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_segment_videos(trial_string, consensus_timepoint, original_folder, 
                         trial_data_dir, sampling_rate_hz=5):
    """
    Create pre/post segment videos using original TIFF folder and consensus timepoint
    """
    print(f"=== CREATING SEGMENT VIDEOS FOR {trial_string} ===")
    
    if not original_folder.exists():
        print(f"  Skipping {original_folder}, does not exist.")
        return

    # Find and sort .tif files - same as make_videos
    tif_files = sorted([f for f in os.listdir(original_folder) if f.endswith(".ome.tif")])
    
    def custom_sort(filename):
        if filename.endswith("_Default.ome.tif"):
            return 0
        elif filename.endswith("_Default_1.ome.tif"):
            return 1
        elif filename.endswith("_Default_2.ome.tif"):
            return 2
        return 3

    tif_files.sort(key=custom_sort)
    
    if not tif_files:
        print(f"  No .tif files found in {original_folder}")
        return
        
    print(f"  Using TIFF files: {tif_files}")
    
    # Load image stack - same as make_videos
    image_stack = tiff.imread(str(original_folder / tif_files[0]))

    if image_stack.ndim != 3:
        print(f"  Skipping {original_folder}, no valid images loaded.")
        return
        
    print(f"  Loaded Z-stack shape: {image_stack.shape}")
    
    # Deinterleave - same as make_videos
    voltage_channel = image_stack[::2, :, :]
    calcium_channel = image_stack[1::2, :, :]
    print(f"  Deinterleaved shapes: V={voltage_channel.shape}, Ca={calcium_channel.shape}")
    
    # Calculate segment boundaries
    pre_start = 200
    pre_end = consensus_timepoint - 200
    post_start = consensus_timepoint + 400
    post_end = voltage_channel.shape[0] - 1
    
    # Create segment videos
    segments_to_create = []
    if pre_start < pre_end and pre_start >= 0:
        segments_to_create.append(('pre', pre_start, pre_end))
    if post_start < post_end and post_start < voltage_channel.shape[0]:
        segments_to_create.append(('post', post_start, min(post_end, voltage_channel.shape[0] - 1)))
    
    for segment_name, start_frame, end_frame in segments_to_create:
        print(f"  Creating {segment_name} segment: frames {start_frame}-{end_frame}")
        
        # Extract segment frames
        voltage_segment = voltage_channel[start_frame:end_frame+1]
        calcium_segment = calcium_channel[start_frame:end_frame+1]
        
        # Apply same enhancement as make_videos
        voltage_enhanced = np.array([canf.adjust_brightness(frame) for frame in voltage_segment])
        calcium_enhanced = np.array([canf.adjust_brightness(frame) for frame in calcium_segment])
        
        # Save segment videos
        voltage_video_file = f"{segment_name}_voltage_{trial_string}.avi"
        calcium_video_file = f"{segment_name}_calcium_{trial_string}.avi"
        
        canf.save_to_avi(voltage_enhanced, voltage_video_file, trial_data_dir)
        canf.save_to_avi(calcium_enhanced, calcium_video_file, trial_data_dir)
        
        print(f"  ✓ Created: {voltage_video_file} & {calcium_video_file}")
 
def create_post_only_videos(trial_string, original_folder, trial_data_dir, sampling_rate_hz=5):
    """
    Create videos for post-only trials (entire recording as one video)
    """
    print(f"=== CREATING POST-ONLY VIDEOS FOR {trial_string} ===")
    
    if not original_folder.exists():
        print(f"  Skipping {original_folder}, does not exist.")
        return

    # Find and sort .tif files - same as create_segment_videos
    tif_files = sorted([f for f in os.listdir(original_folder) if f.endswith(".ome.tif")])
    
    def custom_sort(filename):
        if filename.endswith("_Default.ome.tif"):
            return 0
        elif filename.endswith("_Default_1.ome.tif"):
            return 1
        elif filename.endswith("_Default_2.ome.tif"):
            return 2
        return 3

    tif_files.sort(key=custom_sort)
    
    if not tif_files:
        print(f"  No .tif files found in {original_folder}")
        return
        
    print(f"  Using TIFF files: {tif_files}")
    
    # Load image stack - same as create_segment_videos
    image_stack = tiff.imread(str(original_folder / tif_files[0]))

    if image_stack.ndim != 3:
        print(f"  Skipping {original_folder}, no valid images loaded.")
        return
        
    print(f"  Loaded Z-stack shape: {image_stack.shape}")
    
    # Deinterleave - same as create_segment_videos
    voltage_channel = image_stack[::2, :, :]
    calcium_channel = image_stack[1::2, :, :]
    print(f"  Deinterleaved shapes: V={voltage_channel.shape}, Ca={calcium_channel.shape}")
    
    # For post-only, use the entire recording
    print(f"  Creating post-only videos: entire recording (frames 0-{voltage_channel.shape[0]-1})")
    
    # Apply same enhancement as other video creation
    voltage_enhanced = np.array([canf.adjust_brightness(frame) for frame in voltage_channel])
    calcium_enhanced = np.array([canf.adjust_brightness(frame) for frame in calcium_channel])
    
    # Save as post videos (since this is post-treatment data)
    voltage_video_file = f"post_voltage_{trial_string}.avi"
    calcium_video_file = f"post_calcium_{trial_string}.avi"
    
    canf.save_to_avi(voltage_enhanced, voltage_video_file, trial_data_dir)
    canf.save_to_avi(calcium_enhanced, calcium_video_file, trial_data_dir)
    
    print(f"  ✓ Created post-only videos: {voltage_video_file} & {calcium_video_file}")
           
def combine_all_events(base_results_dir, data_type="voltage", output_filename=None):
    """
    Combine all individual event CSV files into one master file for each data type
    """
    if output_filename is None:
        output_filename = f"all_events_combined_{data_type}_fixed_std_filtered.csv"
    
    # Search for event files in all trial subdirectories
    event_files = []
    for trial_dir in base_results_dir.iterdir():
        if trial_dir.is_dir():
            trial_event_files = list(trial_dir.glob(f"events_{data_type}_*_fixed_std_adaptive_filtered.csv"))
            event_files.extend(trial_event_files)
    
    if not event_files:
        print(f"No {data_type} event files found to combine")
        return None
    
    print(f"Found {len(event_files)} {data_type} event files to combine")
    
    all_dataframes = []
    for file in event_files:
        df = pd.read_csv(file)
        all_dataframes.append(df)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save combined file in base results directory
    combined_path = base_results_dir / output_filename
    combined_df.to_csv(combined_path, index=False)
    
    print(f"Combined {data_type} events file saved: {combined_path}")
    print(f"Total {data_type} events across all experiments: {len(combined_df)}")
    
    # Print summary statistics
    if len(combined_df) > 0:
        print(f"\n{data_type.capitalize()} summary by toxin and segment:")
        summary_table = combined_df.groupby(['toxin', 'segment']).size().unstack(fill_value=0)
        print(summary_table)
    
    return combined_df

def run_summary_analysis(base_results_dir):
    """
    Run summary analysis combining all trials - CSV files only
    Call this AFTER all HPC jobs are complete
    """
    print("="*80)
    print("RUNNING SUMMARY ANALYSIS - COMBINING EVENT FILES")
    print("="*80)
    
    # Combine events for both modalities
    voltage_combined = combine_all_events(base_results_dir, data_type="voltage")
    calcium_combined = combine_all_events(base_results_dir, data_type="calcium")
    
    if voltage_combined is not None or calcium_combined is not None:
        print("\n✓ Summary CSV files created successfully!")
        print(f"Combined files saved in: {base_results_dir}")
    else:
        print("\n⚠ No event data found to combine")