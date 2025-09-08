"""
Simple Command-Line Event QC Tool with Trial Overview
COMPLETE VERSION: Uses pre-generated plots from event_detection_plots folder
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import datetime

class SimpleEventQC:
    def __init__(self, df_metadata, top_dir, swap_coordinates=True):
        """
        Simple QC tool - trial overview first, then detailed event QC if needed
        """
        self.df_metadata = df_metadata
        self.top_dir = Path(top_dir)
        self.swap_coordinates = swap_coordinates
        
        # Pipeline directories
        self.base_results_dir = Path(top_dir, 'analysis', 'results_pipeline')
        self.video_results_dir = Path(top_dir, 'analysis', 'results_profiles')
        
        self.qc_decisions = {}
        
    def find_trials_from_metadata(self):
        """Find complete trials with improved file detection"""
        print("Finding trials from metadata...")
        complete_trials = {}
        
        for idx, trial_row in self.df_metadata.iterrows():
            trial_string = trial_row.trial_string
            trial_pipeline_dir = self.base_results_dir / trial_string
            trial_video_dir = self.video_results_dir / trial_string
            
            if not trial_pipeline_dir.exists() or not trial_video_dir.exists():
                continue

            # Separate events files from timeseries files
            all_voltage_files = list(trial_pipeline_dir.glob("*voltage*.csv"))
            all_calcium_files = list(trial_pipeline_dir.glob("*calcium*.csv"))
            
            voltage_events = [f for f in all_voltage_files if 'events_' in f.name]
            calcium_events = [f for f in all_calcium_files if 'events_' in f.name]
            voltage_timeseries = [f for f in all_voltage_files if 'events_' not in f.name]
            calcium_timeseries = [f for f in all_calcium_files if 'events_' not in f.name]
            
            '''
            print(f"DEBUG: {trial_string} file classification:")
            print(f"  Voltage events: {[f.name for f in voltage_events]}")
            print(f"  Calcium events: {[f.name for f in calcium_events]}")
            print(f"  Voltage timeseries: {[f.name for f in voltage_timeseries]}")
            print(f"  Calcium timeseries: {[f.name for f in calcium_timeseries]}")
            '''
            
            segment_videos = {
                'voltage': {
                    'pre': trial_pipeline_dir / f"pre_voltage_{trial_string}.avi",
                    'post': trial_pipeline_dir / f"post_voltage_{trial_string}.avi",
                    'full': trial_video_dir / "enhanced_voltage_video.avi"
                },
                'calcium': {
                    'pre': trial_pipeline_dir / f"pre_calcium_{trial_string}.avi", 
                    'post': trial_pipeline_dir / f"post_calcium_{trial_string}.avi",
                    'full': trial_video_dir / "enhanced_calcium_video.avi"
                }
            }
            '''
            print(f"  Expected pre voltage: {segment_videos['voltage']['pre']} - exists: {segment_videos['voltage']['pre'].exists()}")
            print(f"  Expected post voltage: {segment_videos['voltage']['post']} - exists: {segment_videos['voltage']['post'].exists()}")
            print(f"  Expected full voltage: {segment_videos['voltage']['full']} - exists: {segment_videos['voltage']['full'].exists()}")
            print(f"  Expected pre calcium: {segment_videos['calcium']['pre']} - exists: {segment_videos['calcium']['pre'].exists()}")
            print(f"  Expected post calcium: {segment_videos['calcium']['post']} - exists: {segment_videos['calcium']['post'].exists()}")
            print(f"  Expected full calcium: {segment_videos['calcium']['full']} - exists: {segment_videos['calcium']['full'].exists()}")
            '''
            has_voltage_video = (segment_videos['voltage']['full'].exists() or 
                               (segment_videos['voltage']['pre'].exists() and segment_videos['voltage']['post'].exists()))
            has_calcium_video = (segment_videos['calcium']['full'].exists() or 
                               (segment_videos['calcium']['pre'].exists() and segment_videos['calcium']['post'].exists()))
            
            if (voltage_events and calcium_events and 
                voltage_timeseries and calcium_timeseries and 
                has_voltage_video and has_calcium_video):
                
                complete_trials[trial_string] = {
                    'voltage_events': voltage_events[0],
                    'calcium_events': calcium_events[0],
                    'voltage_timeseries': voltage_timeseries,
                    'calcium_timeseries': calcium_timeseries,
                    'videos': segment_videos,
                    'trial_row': trial_row
                }
                print(f"  ✓ {trial_string}")
        
        return complete_trials
    
    def deduplicate_events(self, events_df):
        """Remove duplicate events based on key properties"""
        if len(events_df) == 0:
            return events_df
        
        print(f"Checking for duplicate events in {len(events_df)} events...")
        
        # Define columns that make an event unique
        unique_columns = ['cell_index', 'start_time_sec', 'end_time_sec', 'duration_sec', 'amplitude', 'event_type']
        
        # Check if all required columns exist
        missing_cols = [col for col in unique_columns if col not in events_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns for deduplication: {missing_cols}")
            # Use available columns only
            unique_columns = [col for col in unique_columns if col in events_df.columns]
        
        # Count duplicates before removal
        duplicate_mask = events_df.duplicated(subset=unique_columns, keep='first')
        n_duplicates = duplicate_mask.sum()
        
        if n_duplicates > 0:
            print(f"Found {n_duplicates} duplicate events - removing duplicates")
            
            # Show some examples of duplicates
            print("Example duplicates found:")
            duplicates = events_df[duplicate_mask][unique_columns].head(3)
            for idx, dup in duplicates.iterrows():
                print(f"  Cell {dup['cell_index']}: {dup['start_time_sec']:.1f}-{dup['end_time_sec']:.1f}s, "
                    f"amp={dup.get('amplitude', 'N/A'):.3f}, type={dup['event_type']}")
            
            # Remove duplicates (keep first occurrence)
            deduplicated_df = events_df.drop_duplicates(subset=unique_columns, keep='first').reset_index(drop=True)
            
            print(f"Events after deduplication: {len(deduplicated_df)} (removed {n_duplicates} duplicates)")
            return deduplicated_df
        else:
            print("No duplicate events found")
            return events_df
    
    def load_segment_data(self, trial_string, trial_files, segment_name, data_type):
        """Load data for a segment - includes timeseries data"""
        # Load events
        if data_type == 'voltage':
            events_df = pd.read_csv(trial_files['voltage_events'])
            timeseries_files = trial_files['voltage_timeseries']
        else:
            events_df = pd.read_csv(trial_files['calcium_events'])
            timeseries_files = trial_files['calcium_timeseries']
        
        # First deduplicate the entire events dataframe
        events_df = self.deduplicate_events(events_df)
        # Filter by segment
        if 'segment' in events_df.columns:
            segment_events = events_df[events_df['segment'] == segment_name].reset_index(drop=True)
        else:
            segment_events = events_df
        
        # Get video path
        video_file = self.get_best_video_for_segment(trial_files['videos'], data_type, segment_name)
        if video_file is None:
            return None
        
        # Load timeseries
        timeseries_file = self.get_best_timeseries_for_segment(timeseries_files, segment_name)
        if timeseries_file is None:
            return None
            
        timeseries_df = pd.read_csv(timeseries_file)
        '''
         # ADD THIS DETAILED DEBUG:
        print(f"DEBUG: Loading {timeseries_file}")
        print(f"DEBUG: DataFrame shape: {timeseries_df.shape}")
        print(f"DEBUG: Column names (with quotes): {[repr(col) for col in timeseries_df.columns]}")
        print(f"DEBUG: Looking for 'cell_x' and 'cell_y'")
        print(f"DEBUG: 'cell_x' in columns: {'cell_x' in timeseries_df.columns}")
        print(f"DEBUG: 'cell_y' in columns: {'cell_y' in timeseries_df.columns}")
        '''
        # Fix coordinates if needed
        cell_positions = timeseries_df[['cell_x', 'cell_y']].copy()
        if self.swap_coordinates:
            cell_positions['cell_x_corrected'] = timeseries_df['cell_y']
            cell_positions['cell_y_corrected'] = timeseries_df['cell_x']
            cell_positions = cell_positions[['cell_x_corrected', 'cell_y_corrected']].rename(
                columns={'cell_x_corrected': 'cell_x', 'cell_y_corrected': 'cell_y'}
            )
        
        # Load timeseries data efficiently
        data_start_col = 0
        for col in timeseries_df.columns:
            if col in ['cell_id', 'cell_x', 'cell_y']:
                data_start_col = timeseries_df.columns.get_loc(col) + 1
            elif str(col).replace('.', '').isdigit():
                data_start_col = timeseries_df.columns.get_loc(col)
                break
        
        timeseries_data = timeseries_df.iloc[:, data_start_col:]
        
        return {
            'events_df': segment_events,
            'cell_positions': cell_positions,
            'timeseries_data': timeseries_data,
            'video_file': video_file,
            'trial_string': trial_string,
            'segment_name': segment_name,
            'data_type': data_type,
            'sampling_rate': 5
        }
    
    def get_best_video_for_segment(self, videos_dict, data_type, segment_name):
        """Get best video path"""
        video_info = videos_dict[data_type]
        segment_video = video_info[segment_name]
        
        if segment_video.exists():
            return segment_video
        
        full_video = video_info['full']
        if full_video.exists():
            return full_video
        
        return None
    
    def get_best_timeseries_for_segment(self, timeseries_files, segment_name):
        """Get best timeseries file - improved to avoid events files"""
        
        print(f"DEBUG: Looking for {segment_name} timeseries in: {[f.name for f in timeseries_files]}")
        
        # Filter out events files (they contain 'events_' in the name)
        actual_timeseries_files = [f for f in timeseries_files if 'events_' not in f.name.lower()]
        
        print(f"DEBUG: After filtering out events files: {[f.name for f in actual_timeseries_files]}")
        
        # First, try to find exact segment match
        for ts_file in actual_timeseries_files:
            filename = ts_file.name.lower()
            if f'{segment_name}_' in filename:
                print(f"DEBUG: Found exact match: {ts_file.name}")
                return ts_file
        
        # For post-only trials looking for 'post', accept files with 'post_' in name
        if segment_name == 'post':
            for ts_file in actual_timeseries_files:
                if 'post_' in ts_file.name.lower():
                    print(f"DEBUG: Found post file: {ts_file.name}")
                    return ts_file
        
        # Last resort: use first non-events file
        if actual_timeseries_files:
            fallback_file = actual_timeseries_files[0]
            print(f"DEBUG: Using fallback file: {fallback_file.name}")
            return fallback_file
        
        print(f"DEBUG: No timeseries files found for {segment_name}")
        return None
    
    def load_existing_timeseries_plot(self, segment_data):
        """Load pre-generated timeseries plot from event_detection_plots folder"""
        try:
            trial_string = segment_data['trial_string']
            segment_name = segment_data['segment_name']
            data_type = segment_data['data_type']
            
            # Get toxin name from metadata
            toxin = "unknown_toxin"
            if len(self.df_metadata) > 0:
                # Try to get toxin from the trial metadata
                trial_row = self.df_metadata[self.df_metadata['trial_string'] == trial_string]
                if len(trial_row) > 0:
                    toxin = trial_row.iloc[0].get('expt', 'unknown_toxin')
                else:
                    # Fallback to first row's toxin
                    toxin = self.df_metadata['expt'].iloc[0]
            
            # Look for the plot in event_detection_plots folder
            plots_dir = self.base_results_dir / 'event_detection_plots'
            
            if not plots_dir.exists():
                print(f"Event detection plots directory not found: {plots_dir}")
                return None
            
            # Use the correct naming pattern you specified
            plot_filename = f"{segment_name}_{data_type}_{toxin}_{trial_string}_events_perchannel_fixed.png"
            plot_file = plots_dir / plot_filename
            
            # Also try some alternative patterns in case toxin name has variations
            possible_patterns = [
                f"{segment_name}_{data_type}_{toxin}_{trial_string}_events_perchannel_fixed.png",
                f"{segment_name}_{data_type}_{toxin}_{trial_string}_*_events_perchannel_fixed.png",
                f"{segment_name}_{data_type}_*_{trial_string}_events_perchannel_fixed.png"
            ]
            
            plot_file = None
            for pattern in possible_patterns:
                if '*' in pattern:
                    # Use glob for wildcard patterns
                    matching_files = list(plots_dir.glob(pattern))
                    if matching_files:
                        plot_file = matching_files[0]  # Use first match
                        break
                else:
                    # Direct file check
                    test_file = plots_dir / pattern
                    if test_file.exists():
                        plot_file = test_file
                        break
            
            if plot_file is None or not plot_file.exists():
                print(f"No pre-generated plot found for {trial_string} {segment_name} {data_type}")
                print(f"Expected filename: {segment_name}_{data_type}_{toxin}_{trial_string}_events_perchannel_fixed.png")
                print(f"Searched in: {plots_dir}")
                print(f"Toxin used: {toxin}")
                return None
            
            print(f"Loading existing plot: {plot_file.name}")
            
            # Load the image using OpenCV
            plot_image = cv2.imread(str(plot_file))
            
            if plot_image is None:
                print(f"Failed to load image from {plot_file}")
                return None
            
            print(f"Successfully loaded pre-generated plot: {plot_file.name}")
            return plot_image
            
        except Exception as e:
            print(f"Error loading existing timeseries plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def play_trial_overview_video(self, segment_data):
        """Play overview of the entire trial video with pre-generated timeseries plot"""
        video_file = segment_data['video_file']
        trial_string = segment_data['trial_string']
        segment_name = segment_data['segment_name']
        data_type = segment_data['data_type']
        print(f"DEBUG: Loading video from: {video_file}")
        print(f"\nPlaying OVERVIEW video for {trial_string} - {segment_name} - {data_type}")
        print("Check for: floaters, focus issues, movement artifacts, etc.")
        print("Press: a=accept all events, r=reject all events, q=detailed QC, space=replay")
        
        # Load video
        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print("Could not load video for overview")
            return None
        
        # Sample frames for faster overview
        frame_skip = max(1, total_frames // 1000)  # Limit to ~1000 frames max
        
        frames = []
        frame_indices = []
        for i in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            frame_indices.append(i)
        
        cap.release()
        
        if not frames:
            print("No frames loaded for overview")
            return None
        
        # Get cell positions for overlay
        cell_positions = segment_data['cell_positions']
        sampling_rate = segment_data['sampling_rate']
        
        print(f"Overview: {len(frames)} frames (skipping {frame_skip} frames each)")
        print("Loading pre-generated timeseries plot...")
        
        # Load pre-generated timeseries plot
        plot_image = self.load_existing_timeseries_plot(segment_data)
        
        if plot_image is None:
            print("Warning: Could not load pre-generated plot, proceeding with video only")
        else:
            # Resize plot to fit screen height (e.g., max 800 pixels high)
            max_height = 1000  # Adjust this value based on your screen
            height, width = plot_image.shape[:2]
            if height > max_height:
                scale_factor = max_height / height
                new_width = int(width * scale_factor)
                plot_image = cv2.resize(plot_image, (new_width, max_height))
                print(f"Resized plot from {width}x{height} to {new_width}x{max_height}")
        
        while True:
            for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
                # Create display frame
                display_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
                
                # Add cell overlays
                for cell_idx, (x, y) in cell_positions.iterrows():
                    cv2.circle(display_frame, (int(x), int(y)), 12, (0, 255, 255), 2)
                    cv2.putText(display_frame, str(cell_idx), 
                               (int(x)-5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Add timing and progress info
                current_time = frame_idx / sampling_rate
                progress = (i + 1) / len(frames) * 100
                
                cv2.putText(display_frame, f'Overview: {trial_string} - {segment_name} - {data_type}', 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f'Time: {current_time:.1f}s | Progress: {progress:.1f}%', 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f'Frame: {i+1}/{len(frames)} | a=accept all, r=reject all, q=detailed QC', 
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Show pre-generated timeseries plot
                if plot_image is not None:
                    # Add current time indicator to the static plot
                    plot_display = plot_image.copy()
                    
                    # Calculate x-position for time indicator line
                    # Assume the plot shows full trial duration - you may need to adjust this
                    plot_width = plot_display.shape[1]
                    trial_duration = total_frames / sampling_rate  # Total trial time
                    if trial_duration > 0:
                        x_position = int((current_time / trial_duration) * plot_width)
                        # Draw vertical line in bright green
                        cv2.line(plot_display, (x_position, 0), (x_position, plot_display.shape[0]), 
                                (0, 255, 0), 3)
                    
                    cv2.imshow('Timeseries Plot', plot_display)
                    cv2.moveWindow('Timeseries Plot', 50, 5)
                
                # Resize video for better visibility (2x larger)
                display_frame = cv2.resize(display_frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Trial Overview', display_frame)
                cv2.moveWindow('Trial Overview', 700, 5)  # Position video to the right
                
                key = cv2.waitKey(1) & 0xFF  # Fast playback 
                
                if key == ord('a'):
                    cv2.destroyAllWindows()
                    return 'accept_all'
                elif key == ord('r'):
                    cv2.destroyAllWindows()
                    return 'reject_all'
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return 'detailed_qc'
                elif key == ord(' '):  # Spacebar to replay
                    print("Replaying overview...")
                    break
                elif key != 255:  # Any other key pauses
                    cv2.waitKey(0)
            
        cv2.destroyAllWindows()
        return None

    def create_full_cell_timeseries_plot(self, event, segment_data, current_time=None):
        """Create timeseries plot showing FULL cell timeseries with minimal normalization to preserve fluctuations"""
        try:
            if 'timeseries_data' not in segment_data:
                print("No timeseries data available")
                return None
                
            cell_index = event['cell_index']
            sampling_rate = segment_data['sampling_rate']
            
            print(f"Creating FULL timeseries plot for cell {cell_index}...")
            
            # Check if cell index is valid
            if cell_index >= len(segment_data['timeseries_data']):
                print(f"Cell index {cell_index} out of range (max: {len(segment_data['timeseries_data'])-1})")
                return None
            
            # Get FULL timeseries data for this specific cell
            cell_timeseries = segment_data['timeseries_data'].iloc[cell_index].values
            
            # Create full time axis
            time_axis = np.arange(len(cell_timeseries)) / sampling_rate
            
            # Import matplotlib here with proper backend
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            
            # Create figure showing FULL timeseries
            fig, ax = plt.subplots(figsize=(16, 6))
            
            # Minimal normalization - just subtract baseline to preserve real fluctuations
            baseline = np.median(cell_timeseries)
            plot_data = cell_timeseries - baseline
            
            # Plot FULL timeseries in black (like your reference code)
            ax.plot(time_axis, plot_data, 'k-', linewidth=1.0, alpha=0.8, label='Full Cell Timeseries')
            
            # Add baseline line (green like your reference)
            ax.axhline(y=0, color='green', linestyle='-', linewidth=1.5, alpha=0.8, label='Baseline (median)')
            
            # Highlight ALL events for this cell (for context)
            events_df = segment_data['events_df']
            cell_events = events_df[events_df['cell_index'] == cell_index]
            
            for idx, other_event in cell_events.iterrows():
                event_start = other_event['start_time_sec']
                event_end = other_event['end_time_sec']
                
                if idx == event.name:  # Current event being reviewed
                    # Highlight current event prominently
                    event_color = 'red' if other_event['event_type'] == 'positive' else 'blue'
                    ax.axvspan(event_start, event_end, alpha=0.6, color=event_color,
                              label=f"CURRENT EVENT (#{idx}): {other_event['event_type'].title()}")
                    
                    # Add thick vertical lines for current event boundaries
                    ax.axvline(event_start, color=event_color, linestyle='-', linewidth=3, alpha=0.9)
                    ax.axvline(event_end, color=event_color, linestyle='-', linewidth=3, alpha=0.9)
                    
                    # Add event markers (like your reference code)
                    start_sample = int(event_start * sampling_rate)
                    end_sample = int(event_end * sampling_rate)
                    if start_sample < len(cell_timeseries) and end_sample < len(cell_timeseries):
                        y_start = cell_timeseries[start_sample] - baseline
                        y_end = cell_timeseries[end_sample] - baseline
                        ax.plot(event_start, y_start, 'o', color=event_color, markersize=6, alpha=0.9)
                        ax.plot(event_end, y_end, 's', color=event_color, markersize=6, alpha=0.9)
                    
                    # Add event number label
                    mid_time = (event_start + event_end) / 2
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    label_y = ax.get_ylim()[1] - y_range * 0.1
                    ax.text(mid_time, label_y, f'Event #{idx}', 
                           ha='center', va='center', fontsize=12, weight='bold', 
                           color=event_color, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                else:
                    # Show other events for context (lighter)
                    other_color = 'pink' if other_event['event_type'] == 'positive' else 'lightblue'
                    ax.axvspan(event_start, event_end, alpha=0.2, color=other_color)
            
            # Add current time indicator (for video sync)
            if current_time is not None:
                ax.axvline(current_time, color='lime', linestyle='-', linewidth=3, alpha=0.9, 
                          label=f'Current Video Time: {current_time:.1f}s')
                
                # Add text indicator for current time
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                label_y = ax.get_ylim()[0] + y_range * 0.9
                ax.text(current_time, label_y, f'{current_time:.1f}s', 
                       ha='center', va='center', fontsize=10, weight='bold', 
                       color='lime', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
            
            # Formatting
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Signal (baseline subtracted)', fontsize=12)
            
            title_text = f'Cell {cell_index} - {segment_data["data_type"].title()} - FULL Timeseries\n'
            title_text += f'Current Event #{event.name}: {event["start_time_sec"]:.1f}-{event["end_time_sec"]:.1f}s '
            title_text += f'({event["event_type"]})'
            
            if current_time is not None:
                title_text += f' | Video Time: {current_time:.1f}s'
            
            ax.set_title(title_text, fontsize=13, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Set x-axis to show full time range
            ax.set_xlim(0, time_axis[-1])
            
            plt.tight_layout()
            
            # Convert to OpenCV image
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Convert RGB to BGR for OpenCV
            plot_image = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            
            return plot_image
            
        except Exception as e:
            print(f"Error creating full cell timeseries plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def show_event_info(self, event, event_idx, segment_data):
        """Print event information"""
        print(f"\n{'='*50}")
        print(f"EVENT {event_idx}")
        print(f"{'='*50}")
        print(f"Trial: {segment_data['trial_string']}")
        print(f"Segment: {segment_data['segment_name'].upper()}")
        print(f"Data Type: {segment_data['data_type'].upper()}")
        print(f"Cell: {event['cell_index']}")
        print(f"Time: {event['start_time_sec']:.1f} - {event['end_time_sec']:.1f} seconds")
        print(f"Duration: {event['end_time_sec'] - event['start_time_sec']:.1f} seconds")
        print(f"Type: {event['event_type']}")
        if 'amplitude' in event:
            print(f"Amplitude: {event['amplitude']:.3f}")
        print(f"{'='*50}")
    
    def play_event_video(self, event, segment_data):
        """Play video clip for an event with FULL cell timeseries plot and current time indicator"""
        # Calculate video bounds (3 seconds before to 3 seconds after)
        start_time = max(0, event['start_time_sec'] - 3.0)
        end_time = event['end_time_sec'] + 3.0
        
        start_frame = int(start_time * segment_data['sampling_rate'])
        end_frame = int(end_time * segment_data['sampling_rate'])
        
        # Load video frames
        cap = cv2.VideoCapture(str(segment_data['video_file']))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            print("No video frames available for this event")
            return None
        
        # Get cell position
        cell_x = int(segment_data['cell_positions'].iloc[event['cell_index']]['cell_x'])
        cell_y = int(segment_data['cell_positions'].iloc[event['cell_index']]['cell_y'])
        
        print(f"Playing video clip ({len(frames)} frames)")
        print("Full cell timeseries will be shown with current video time indicator")
        print("Press: a=accept, r=reject, s=skip, space=replay, any other key=pause")
        
        while True:
            for i, frame in enumerate(frames):
                # Create display frame
                display_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
                
                # Highlight the cell
                cv2.circle(display_frame, (cell_x, cell_y), 20, (0, 255, 255), 3)
                cv2.putText(display_frame, f'Cell {event["cell_index"]}', 
                           (cell_x + 25, cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Calculate current time in the full timeseries
                current_time = start_time + i / segment_data['sampling_rate']
                event_active = event['start_time_sec'] <= current_time <= event['end_time_sec']
                
                if event_active:
                    status_color = (0, 255, 0)  # Green during event
                    status_text = "EVENT ACTIVE"
                else:
                    status_color = (255, 255, 255)  # White outside event
                    status_text = "BASELINE"
                
                cv2.putText(display_frame, f'Time: {current_time:.1f}s | {status_text}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(display_frame, f'Frame: {i+1}/{len(frames)} | a=accept, r=reject, s=skip', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                '''
                # Create updated timeseries plot with current time indicator
                updated_plot = self.create_full_cell_timeseries_plot(event, segment_data, current_time)
                
                if updated_plot is not None:
                    cv2.imshow('Full Cell Timeseries', updated_plot)
                    cv2.moveWindow('Full Cell Timeseries', 50, 5)
                '''
                # Load static timeseries plot once (like in trial overview)
                if i == 0:  # Only load once at the beginning
                    static_plot = self.load_existing_timeseries_plot(segment_data)
                    if static_plot is not None:
                        # Apply same resize logic as in trial overview
                        max_height = 1000
                        height, width = static_plot.shape[:2]
                        if height > max_height:
                            scale_factor = max_height / height
                            new_width = int(width * scale_factor)
                            static_plot = cv2.resize(static_plot, (new_width, max_height))

                # Show static plot with moving vertical line (like trial overview)
                if 'static_plot' in locals() and static_plot is not None:
                    plot_display = static_plot.copy()
                    
                    # Calculate x-position for time indicator line (same as trial overview)
                    plot_width = plot_display.shape[1]
                    trial_duration = total_frames / segment_data['sampling_rate']
                    if trial_duration > 0:
                        x_position = int((current_time / trial_duration) * plot_width)
                        # Draw vertical line in bright green
                        cv2.line(plot_display, (x_position, 0), (x_position, plot_display.shape[0]), 
                                (0, 255, 0), 3)
                    
                    cv2.imshow('Static Timeseries', plot_display)
                    cv2.moveWindow('Static Timeseries', 50, 5)

                # Resize video for better visibility (2x larger)
                display_frame = cv2.resize(display_frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                # Position video window next to timeseries
                cv2.imshow('Event Video', display_frame)
                cv2.moveWindow('Event Video', 700, 5)
                
                key = cv2.waitKey(10) & 0xFF  # Fast playback 
                
                # Streamlined controls - make decision during video
                if key == ord('a'):
                    cv2.destroyAllWindows()
                    return 'accept'
                elif key == ord('r'):
                    cv2.destroyAllWindows()
                    return 'reject'
                elif key == ord('s'):
                    cv2.destroyAllWindows()
                    return 'skip'
                elif key == ord(' '):  # Spacebar to replay
                    print("Replaying...")
                    break
                elif key != 255:  # Any other key pauses
                    cv2.waitKey(0)  # Wait for another key press
            
        cv2.destroyAllWindows()
        return None
    
    def review_single_event(self, event, event_idx, segment_data):
        """Review a single event - automatically plays video with direct decision making"""
        self.show_event_info(event, event_idx, segment_data)
        
        # Automatically play video clip and get decision
        print("\nPlaying video clip automatically...")
        print("While watching: press a=accept, r=reject, s=skip")
        decision = self.play_event_video(event, segment_data)
        
        if decision:
            if decision == 'accept':
                print("✓ Event ACCEPTED")
            elif decision == 'reject':
                print("✗ Event REJECTED")
            elif decision == 'skip':
                print("? Event SKIPPED")
            return decision
        
        # If no decision was made during video, ask for decision
        while True:
            print("\nOptions:")
            print("  a - Accept this event")
            print("  r - Reject this event")
            print("  s - Skip this event (decide later)")
            print("  v - View video clip again")
            print("  i - Show event info again")
            
            choice = input("Your choice: ").strip().lower()
            
            if choice == 'a':
                print("✓ Event ACCEPTED")
                return 'accept'
            elif choice == 'r':
                print("✗ Event REJECTED")
                return 'reject'
            elif choice == 's':
                print("? Event SKIPPED")
                return 'skip'
            elif choice == 'v':
                decision = self.play_event_video(event, segment_data)
                if decision:
                    if decision == 'accept':
                        print("✓ Event ACCEPTED")
                    elif decision == 'reject':
                        print("✗ Event REJECTED")
                    elif decision == 'skip':
                        print("? Event SKIPPED")
                    return decision
            elif choice == 'i':
                self.show_event_info(event, event_idx, segment_data)
            else:
                print("Invalid choice. Please use a/r/s/v/i")
    
    def qc_segment(self, segment_data):
        """QC all events in a segment - with trial overview first"""
        events_df = segment_data['events_df']
        
        if len(events_df) == 0:
            print("No events to review in this segment")
            return {'decision': 'accept', 'rejected_events': []}
        
        print(f"\n{'='*60}")
        print(f"SEGMENT QC: {segment_data['trial_string']}")
        print(f"Segment: {segment_data['segment_name'].upper()}")
        print(f"Data Type: {segment_data['data_type'].upper()}")
        print(f"Total Events: {len(events_df)}")
        print(f"{'='*60}")
        
        # Show event summary
        print("\nEvent Summary:")
        for idx, event in events_df.iterrows():
            print(f"  Event {idx}: Cell {event['cell_index']}, "
                  f"{event['start_time_sec']:.1f}-{event['end_time_sec']:.1f}s, "
                  f"Type: {event['event_type']}")
        
        # NEW: Trial overview first
        print(f"\n{'='*50}")
        print("STEP 1: TRIAL OVERVIEW")
        print(f"{'='*50}")
        print("First, let's check the overall video quality for this trial.")
        
        overview_decision = self.play_trial_overview_video(segment_data)
        
        if overview_decision == 'accept_all':
            print("✓ ACCEPTED ALL EVENTS based on trial overview")
            return {'decision': 'accept', 'rejected_events': []}
        
        elif overview_decision == 'reject_all':
            print("✗ REJECTED ALL EVENTS based on trial overview")
            return {'decision': 'reject', 'rejected_events': list(events_df.index)}
        
        elif overview_decision == 'detailed_qc':
            print("→ Proceeding to detailed event-by-event QC")
        
        else:
            # If no decision was made, ask what to do
            print("\nTrial overview completed. What would you like to do?")
            print("  a - Accept all events (good quality)")
            print("  r - Reject all events (poor quality)")
            print("  d - Detailed event-by-event QC")
            
            choice = input("Your choice: ").strip().lower()
            
            if choice == 'a':
                print("✓ ACCEPTED ALL EVENTS")
                return {'decision': 'accept', 'rejected_events': []}
            elif choice == 'r':
                print("✗ REJECTED ALL EVENTS")
                return {'decision': 'reject', 'rejected_events': list(events_df.index)}
            elif choice == 'd':
                print("→ Proceeding to detailed event-by-event QC")
            else:
                print("Invalid choice, proceeding to detailed QC")
        
        # Continue with detailed event-by-event QC
        print(f"\n{'='*50}")
        print("STEP 2: DETAILED EVENT QC")
        print(f"{'='*50}")
        
        event_decisions = {}
        current_event = 0
        
        while current_event < len(events_df):
            event = events_df.iloc[current_event]
            event_idx = events_df.index[current_event]
            
            print(f"\n{'='*40}")
            print(f"Reviewing Event {current_event + 1} of {len(events_df)}")
            print(f"{'='*40}")
            
            # Check if we have a decision for this event already
            if event_idx in event_decisions:
                print(f"Previous decision: {event_decisions[event_idx]}")
                print("Options:")
                print("  n - Next event")
                print("  p - Previous event")
                print("  c - Change decision")
                print("  q - Quit this segment")
                
                choice = input("Your choice: ").strip().lower()
                
                if choice == 'n':
                    current_event += 1
                elif choice == 'p':
                    current_event = max(0, current_event - 1)
                elif choice == 'c':
                    decision = self.review_single_event(event, event_idx, segment_data)
                    if decision != 'skip':
                        event_decisions[event_idx] = decision
                    # Automatically go to next event after decision
                    current_event += 1
                elif choice == 'q':
                    break
                else:
                    print("Invalid choice")
            else:
                # Review this event
                decision = self.review_single_event(event, event_idx, segment_data)
                
                if decision != 'skip':
                    event_decisions[event_idx] = decision
                
                # Automatically proceed to next event after accept/reject
                current_event += 1
                
                # Only ask for navigation if user wants to go back or quit
                if current_event < len(events_df):
                    print(f"\nAutomatically proceeding to next event ({current_event + 1}/{len(events_df)})")
                    print("Press Enter to continue, or type:")
                    print("  p - Go back to previous event")
                    print("  q - Quit this segment")
                    
                    next_choice = input("Choice (or Enter): ").strip().lower()
                    
                    if next_choice == 'p':
                        current_event = max(0, current_event - 1)
                    elif next_choice == 'q':
                        break
                    # Otherwise continue automatically (Enter or any other input)
        
        # Process decisions
        rejected_events = [idx for idx, decision in event_decisions.items() if decision == 'reject']
        
        print(f"\nFinal Summary:")
        print(f"Total events: {len(events_df)}")
        print(f"Reviewed events: {len(event_decisions)}")
        print(f"Accepted events: {len([d for d in event_decisions.values() if d == 'accept'])}")
        print(f"Rejected events: {len(rejected_events)}")
        print(f"Unreviewed events: {len(events_df) - len(event_decisions)}")
        
        # Ask about unreviewed events
        unreviewed_count = len(events_df) - len(event_decisions)
        if unreviewed_count > 0:
            print(f"\nWhat to do with {unreviewed_count} unreviewed events?")
            print("  a - Accept all unreviewed")
            print("  r - Reject all unreviewed")
            print("  k - Keep as unreviewed (will be accepted by default)")
            
            choice = input("Your choice: ").strip().lower()
            
            if choice == 'r':
                # Reject all unreviewed
                for idx in events_df.index:
                    if idx not in event_decisions:
                        rejected_events.append(idx)
        
        # Determine final decision
        if len(rejected_events) == 0:
            final_decision = 'accept'
        elif len(rejected_events) == len(events_df):
            final_decision = 'reject'
        else:
            final_decision = 'partial_reject'
        
        return {
            'decision': final_decision,
            'rejected_events': rejected_events
        }
    
    def is_trial_complete(self, trial_string):
        """Check if trial has all required final QC files (handles post-only trials)"""
        trial_dir = self.base_results_dir / trial_string
        
        # Detect available segments for this trial (reuse the logic)
        trial_files = self.find_trials_from_metadata().get(trial_string)
        if not trial_files:
            return False
        
        available_segments = self.detect_available_segments(trial_string, trial_files)
        
        # Check for required files based on available segments
        required_files = []
        for segment in available_segments:
            for data_type in ['voltage', 'calcium']:
                required_files.append(
                    trial_dir / f"events_{data_type}_{segment}_{trial_string}_simple_QC_final.csv"
                )
        '''
        print(f"DEBUG: Checking completion for {trial_string}")
        print(f"  Available segments: {available_segments}")
        print(f"  Required files: {[f.name for f in required_files]}")
        print(f"  Files exist: {[f.exists() for f in required_files]}")
        '''
        # Check if all required files exist
        return all(file.exists() for file in required_files)
    
    def apply_single_trial_decisions(self, trial_string):
        """Apply QC decisions for a single completed trial - save 4 separate CSV files"""
        if not self.qc_decisions:
            return
        
        # Filter decisions for this trial only
        trial_decisions = {k: v for k, v in self.qc_decisions.items() 
                        if v['trial_string'] == trial_string}
        
        if not trial_decisions:
            return
        
        print(f"Applying QC decisions for {trial_string}...")
        
        trial_dir = self.base_results_dir / trial_string
        
        # Process each data type
        for data_type in ['voltage', 'calcium']:
            source_files = list(trial_dir.glob(f"events_{data_type}_*_filtered.csv"))
            if not source_files:
                continue
                
            source_file = source_files[0]
            events_df = pd.read_csv(source_file)
            
            # Process each segment separately to create individual files
            for segment in ['pre', 'post']:
                qc_key = f"{trial_string}_{segment}_{data_type}"
                if qc_key in trial_decisions:
                    decision_info = trial_decisions[qc_key]
                    decision = decision_info['decision']
                    rejected_events = decision_info.get('rejected_events', [])
                    
                    if 'segment' in events_df.columns:
                        segment_events = events_df[events_df['segment'] == segment]
                    else:
                        segment_events = events_df
                    
                    if decision == 'accept':
                        final_segment_events = segment_events
                    elif decision == 'partial_reject':
                        final_segment_events = segment_events[~segment_events.index.isin(rejected_events)]
                    elif decision == 'reject':
                        final_segment_events = segment_events.iloc[0:0].copy()
                    
                    # Save separate file for each segment
                    segment_file = trial_dir / f"events_{data_type}_{segment}_{trial_string}_simple_QC_final.csv"
                    final_segment_events.to_csv(segment_file, index=False)
                    print(f"✓ Created: {segment_file.name}")
    
    def create_final_toxin_summary(self):
        """Create final toxin summary from all completed trials"""
        # Get all completed trials
        completed_trials = [t for t in self.find_trials_from_metadata().keys() 
                           if self.is_trial_complete(t)]
        
        if not completed_trials:
            print("No completed trials found for summary")
            return
        
        # Create combined summary using existing method logic
        individual_trial_results = {}
        
        for trial_string in completed_trials:
            individual_trial_results[trial_string] = {}
            trial_dir = self.base_results_dir / trial_string
            
            for data_type in ['voltage', 'calcium']:
                final_file = trial_dir / f"events_{data_type}_{trial_string}_simple_QC_final.csv"
                if final_file.exists():
                    individual_trial_results[trial_string][data_type] = pd.read_csv(final_file)
        
        self.create_toxin_summary_files(individual_trial_results)
    
    def run_simple_qc(self):
        """Main QC workflow with trial overview"""
        print("ENHANCED EVENT QC TOOL - WITH PRE-GENERATED TIMESERIES PLOTS")
        print("="*70)
        print("STEP 1: Trial overview (loads pre-generated plots from event_detection_plots)")
        print("STEP 2: Detailed event QC (shows FULL CELL timeseries with video sync)")
        print("="*70)
        
        trials = self.find_trials_from_metadata()
        
        if not trials:
            print("No complete trials found!")
            return
        
        print(f"Found {len(trials)} complete trials")
        
        # Check for already completed trials
        completed_trials = []
        remaining_trials = []
        
        for trial_string in trials.keys():
            if self.is_trial_complete(trial_string):
                completed_trials.append(trial_string)
            else:
                remaining_trials.append(trial_string)
        
        print(f"Already completed: {len(completed_trials)} trials")
        print(f"Remaining to QC: {len(remaining_trials)} trials")
        
        if completed_trials:
            print("Completed trials:")
            for trial in completed_trials:
                print(f"  ✓ {trial}")
        
        if not remaining_trials:
            print("All trials already completed!")
            return
        
        segment_count = 0
        total_segments = len(remaining_trials) * 2 * 2
        
        for trial_string in remaining_trials:
            trial_files = trials[trial_string]
            print(f"\n{'='*80}")
            print(f"PROCESSING TRIAL: {trial_string}")
            print(f"{'='*80}")
            
            trial_completed = True  # Track if all segments for this trial are completed
            
            # Detect available segments for this trial
            available_segments = self.detect_available_segments(trial_string, trial_files)
            print(f"Available segments for {trial_string}: {available_segments}")

            total_segments_for_trial = len(available_segments) * 2  # voltage + calcium

            for segment_name in available_segments:  # Use detected segments instead of hardcoded ['pre', 'post']
                for data_type in ['voltage', 'calcium']:
                    segment_count += 1
                    
                    print(f"\n{'='*60}")
                    print(f"SEGMENT {segment_count}/{total_segments}")
                    print(f"{'='*60}")
                    
                    try:
                        segment_data = self.load_segment_data(trial_string, trial_files, segment_name, data_type)
                        
                        if segment_data is None:
                            print(f"Failed to load {trial_string} {segment_name} {data_type}")
                            trial_completed = False
                            continue
                        
                        # Debug: Check if timeseries data was loaded
                        if 'timeseries_data' in segment_data:
                            print(f"Timeseries data loaded: {segment_data['timeseries_data'].shape}")
                        else:
                            print("WARNING: No timeseries data loaded!")
                        
                        # QC this segment (with trial overview first)
                        qc_result = self.qc_segment(segment_data)
                        
                        # Save decision
                        qc_key = f"{trial_string}_{segment_name}_{data_type}"
                        self.qc_decisions[qc_key] = {
                            'trial_string': trial_string,
                            'segment': segment_name,
                            'data_type': data_type,
                            'decision': qc_result['decision'],
                            'rejected_events': qc_result['rejected_events']
                        }
                        
                    except Exception as e:
                        print(f"Error processing segment: {e}")
                        import traceback
                        traceback.print_exc()
                        trial_completed = False
                        continue
            
            # After completing all segments for this trial, create checkpoint
            if trial_completed:
                print(f"\n{'='*60}")
                print(f"COMPLETED TRIAL: {trial_string}")
                print(f"{'='*60}")
                
                # Apply decisions immediately for this trial
                self.apply_single_trial_decisions(trial_string)
                
                # Save overall QC results
                self.save_qc_results()
                
                print(f"✓ Checkpoint saved for {trial_string}")
            else:
                print(f"✗ Trial {trial_string} incomplete - will retry next time")
        
        print("\n" + "="*80)
        print("QC SESSION COMPLETE")
        print("="*80)
        
        # Final summary
        final_completed = [t for t in trials.keys() if self.is_trial_complete(t)]
        final_remaining = [t for t in trials.keys() if not self.is_trial_complete(t)]
        
        print(f"Total completed trials: {len(final_completed)}")
        print(f"Total remaining trials: {len(final_remaining)}")
        
        if final_remaining:
            print("Still need to QC:")
            for trial in final_remaining:
                print(f"  - {trial}")
        else:
            print("🎉 ALL TRIALS COMPLETED!")
            
            # Create final toxin summary
            apply_now = input("\nCreate final toxin summary files? (y/n): ").strip().lower()
            if apply_now in ['y', 'yes']:
                self.create_final_toxin_summary()
    
    def detect_available_segments(self, trial_string, trial_files):
        """Detect which segments (pre/post) are available for this trial"""
        available_segments = set()
        
        # Check voltage events to see what segments exist
        if 'voltage_events' in trial_files:
            voltage_events_df = pd.read_csv(trial_files['voltage_events'])
            if 'segment' in voltage_events_df.columns:
                available_segments.update(voltage_events_df['segment'].unique())
            else:
                # If no segment column, assume post-only
                available_segments.add('post')
        
        # Also check timeseries files to confirm
        voltage_timeseries = trial_files.get('voltage_timeseries', [])
        for ts_file in voltage_timeseries:
            filename = ts_file.name.lower()
            if 'pre_' in filename:
                available_segments.add('pre')
            elif 'post_' in filename:
                available_segments.add('post')
        
        # Return sorted list for consistency
        segment_order = ['pre', 'post']
        return [seg for seg in segment_order if seg in available_segments]
        
    def save_qc_results(self):
        """Save QC decisions"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.base_results_dir / f"simple_qc_decisions_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.qc_decisions, f, indent=2, default=str)
        
        print(f"QC decisions saved to: {results_file}")
    
    def apply_qc_decisions(self):
        """Apply QC decisions to create final files including toxin summaries"""
        if not self.qc_decisions:
            print("No QC decisions to apply.")
            return
        
        print("Applying QC decisions...")
        
        # Group by trial and data type
        trial_decisions = {}
        for qc_key, decision_info in self.qc_decisions.items():
            trial_string = decision_info['trial_string']
            data_type = decision_info['data_type']
            segment = decision_info['segment']
            
            if trial_string not in trial_decisions:
                trial_decisions[trial_string] = {}
            if data_type not in trial_decisions[trial_string]:
                trial_decisions[trial_string][data_type] = {}
            
            trial_decisions[trial_string][data_type][segment] = decision_info
        
        # Apply decisions for individual trials
        individual_trial_results = {}
        
        for trial_string, trial_data in trial_decisions.items():
            individual_trial_results[trial_string] = {}
            
            for data_type, segment_decisions in trial_data.items():
                
                trial_dir = self.base_results_dir / trial_string
                
                if data_type == 'voltage':
                    source_files = list(trial_dir.glob("events_voltage_*_filtered.csv"))
                else:
                    source_files = list(trial_dir.glob("events_calcium_*_filtered.csv"))
                
                if not source_files:
                    continue
                    
                source_file = source_files[0]
                events_df = pd.read_csv(source_file)
                
                final_events_list = []
                
                for segment in ['pre', 'post']:
                    if segment in segment_decisions:
                        decision_info = segment_decisions[segment]
                        decision = decision_info['decision']
                        rejected_events = decision_info.get('rejected_events', [])
                        
                        if 'segment' in events_df.columns:
                            segment_events = events_df[events_df['segment'] == segment]
                        else:
                            segment_events = events_df
                        
                        if decision == 'accept':
                            final_segment_events = segment_events
                        elif decision == 'partial_reject':
                            final_segment_events = segment_events[~segment_events.index.isin(rejected_events)]
                        elif decision == 'reject':
                            final_segment_events = segment_events.iloc[0:0].copy()
                        
                        final_events_list.append(final_segment_events)
                    else:
                        if 'segment' in events_df.columns:
                            segment_events = events_df[events_df['segment'] == segment]
                        else:
                            segment_events = events_df
                        final_events_list.append(segment_events)
                
                if final_events_list:
                    final_events = pd.concat(final_events_list, ignore_index=True)
                else:
                    final_events = events_df.iloc[0:0].copy()
                
                # Save individual trial file
                final_file = trial_dir / f"events_{data_type}_{trial_string}_simple_QC_final.csv"
                final_events.to_csv(final_file, index=False)
                
                print(f"✓ {final_file.name}: {len(final_events)} events")
                
                # Store for toxin summary
                individual_trial_results[trial_string][data_type] = final_events
        
        # Create toxin-level summary files
        self.create_toxin_summary_files(individual_trial_results)
        
        print("QC complete!")

    def create_toxin_summary_files(self, individual_trial_results):
        """Create combined summary files for all trials of this toxin"""
        print("\nCreating toxin-level summary files...")
        
        # Determine toxin name from metadata
        if len(self.df_metadata) > 0:
            toxin = self.df_metadata['expt'].iloc[0]
        else:
            toxin = "unknown_toxin"
        
        print(f"Toxin: {toxin}")
        
        # Combine all trials for each data type
        for data_type in ['voltage', 'calcium']:
            all_events_list = []
            
            for trial_string, trial_results in individual_trial_results.items():
                if data_type in trial_results:
                    trial_events = trial_results[data_type].copy()
                    
                    # Add trial identifier to each event
                    trial_events['trial_string'] = trial_string
                    
                    # Add trial metadata if available
                    trial_row = self.df_metadata[self.df_metadata['trial_string'] == trial_string]
                    if len(trial_row) > 0:
                        trial_info = trial_row.iloc[0]
                        trial_events['date'] = trial_info.get('date', '')
                        trial_events['area'] = trial_info.get('area', '')
                        trial_events['toxin'] = trial_info.get('expt', toxin)
                        trial_events['concentration'] = trial_info.get('concentration', '')
                    
                    all_events_list.append(trial_events)
            
            if all_events_list:
                # Combine all trials
                combined_events = pd.concat(all_events_list, ignore_index=True)
                
                # Save toxin summary file
                summary_file = self.base_results_dir / f"events_{data_type}_{toxin}_QC_final.csv"
                combined_events.to_csv(summary_file, index=False)
                
                print(f"✓ TOXIN SUMMARY: {summary_file.name}")
                print(f"   Total events: {len(combined_events)}")
                print(f"   Trials included: {len(all_events_list)}")
                
                # Print summary statistics
                if len(combined_events) > 0:
                    print(f"   Events by trial:")
                    trial_counts = combined_events.groupby('trial_string').size()
                    for trial, count in trial_counts.items():
                        print(f"     {trial}: {count} events")
                    
                    if 'segment' in combined_events.columns:
                        print(f"   Events by segment:")
                        segment_counts = combined_events.groupby('segment').size()
                        for segment, count in segment_counts.items():
                            print(f"     {segment}: {count} events")
            else:
                print(f"✗ No {data_type} events found across trials")
        
        print(f"\nToxin summary files created for: {toxin}")


def main():
    """Main function"""
    print("ENHANCED EVENT QC TOOL - WITH PRE-GENERATED TIMESERIES PLOTS")
    print("Command-line interface with:")
    print("  • Trial overview using pre-generated plots from event_detection_plots")
    print("  • Individual event QC showing FULL CELL timeseries with video sync")
    
    # Setup paths
    home = Path.home()
    cell_line = 'MDA_MB_468'
    date = '20250827'
    if "ys5320" in str(home):
        top_dir = Path(home, "firefly_link/Calcium_Voltage_Imaging", f'{cell_line}')
        df_file = Path(top_dir, 'analysis', 'dataframes', f'long_acqs_{cell_line}_all_before_{date}.csv')
    else:
        top_dir = Path(r"R:\home\firefly_link\Calcium_Voltage_Imaging", f'{cell_line}')
        df_file = Path(top_dir, 'analysis', 'dataframes', f'long_acqs_{cell_line}_all_before_{date}.csv')
    
    # Load metadata
    if not df_file.exists():
        print(f"Metadata file not found: {df_file}")
        return
    
    df_raw = pd.read_csv(df_file)
    df_filtered = df_raw[df_raw['multi_tif'] > 1]
    df_filtered = df_filtered[df_filtered['use'] == 'y']
    #df_filtered = df_filtered[df_filtered['expt'] == 'TRAM-34_1uM']
    df_filtered = df_filtered[df_filtered['expt'].str.contains('TRAM-34_1uM', na=False)]
    df_filtered = df_filtered.reset_index(drop=True)
    
    if len(df_filtered) == 0:
        print("No trials found after filtering")
        return
    
    print(f"Found {len(df_filtered)} trials to QC")
    
    # Coordinate swapping
    swap_choice = input("Swap coordinates to fix colleague's mistake? (y/n): ").strip().lower()
    swap_coordinates = swap_choice in ['y', 'yes']
    
    # Create QC tool
    qc_tool = SimpleEventQC(df_filtered, top_dir, swap_coordinates=swap_coordinates)
    
    # Run QC
    qc_tool.run_simple_qc()


if __name__ == "__main__":
    main()