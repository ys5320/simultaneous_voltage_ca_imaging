"""
Updated Interactive Event Quality Control Tool
Integrated with new pipeline structure - uses metadata and segment column filtering
"""

import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
import tifffile as tiff
import json
import datetime
import threading
import time

class UpdatedInteractiveEventQC:
    def __init__(self, df_metadata, top_dir):
        """
        Initialize QC tool with metadata and paths matching your pipeline
        
        Args:
            df_metadata: Filtered DataFrame from your metadata CSV
            top_dir: Base directory (same as in your pipeline)
        """
        self.df_metadata = df_metadata
        self.top_dir = Path(top_dir)
        
        # Pipeline directories (match your create_paper_data.py structure)
        self.base_results_dir = Path(top_dir, 'analysis', 'results_pipeline')
        self.video_results_dir = Path(top_dir, 'analysis', 'results_profiles') 
        self.data_dir = Path(top_dir.parent.parent, 'ca_voltage_imaging_working', 'results')
        
        self.qc_decisions = {}
        
        # Video playback state
        self.current_frame = 0
        self.playing = False
        self.auto_play_active = False
        self.event_play_active = False
        self.zoom_factor = 1.0
        self.zoom_center = (256, 256)
        
        # Event selection state
        self.selected_events = set()
        self.selected_cells = set()
        
        # UI elements
        self.fig = None
        self.ax_video = None
        self.ax_timeseries = None
        self.current_trial_data = None
        
    def find_trials_from_metadata(self):
        """
        Find trials using metadata CSV - uses segment column for filtering
        """
        print("Finding trials from metadata...")
        complete_trials = {}
        
        for idx, trial_row in self.df_metadata.iterrows():
            trial_string = trial_row.trial_string
            print(f"Checking trial: {trial_string}")
            
            # Check for pipeline outputs
            trial_pipeline_dir = self.base_results_dir / trial_string
            trial_video_dir = self.video_results_dir / trial_string
            
            if not trial_pipeline_dir.exists() or not trial_video_dir.exists():
                print(f"  ✗ Missing directories")
                continue
            
            # Look for event files (voltage and calcium)
            voltage_events = list(trial_pipeline_dir.glob("events_voltage_*_filtered.csv"))
            calcium_events = list(trial_pipeline_dir.glob("events_calcium_*_filtered.csv"))
            
            # Look for processed timeseries files
            voltage_timeseries = list(trial_pipeline_dir.glob("*_voltage_*.csv"))
            calcium_timeseries = list(trial_pipeline_dir.glob("*_calcium_*.csv"))
            
            # Look for videos (try segment videos first, then fall back to full videos)
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
            
            # Check completeness
            has_voltage_video = (segment_videos['voltage']['full'].exists() or 
                               (segment_videos['voltage']['pre'].exists() and segment_videos['voltage']['post'].exists()))
            has_calcium_video = (segment_videos['calcium']['full'].exists() or 
                               (segment_videos['calcium']['pre'].exists() and segment_videos['calcium']['post'].exists()))
            
            if (voltage_events and calcium_events and 
                voltage_timeseries and calcium_timeseries and 
                has_voltage_video and has_calcium_video):
                
                print(f"  ✓ COMPLETE TRIAL FOUND:")
                print(f"    Voltage events: {voltage_events[0].name}")
                print(f"    Calcium events: {calcium_events[0].name}")
                print(f"    Voltage timeseries: {len(voltage_timeseries)} files")
                print(f"    Calcium timeseries: {len(calcium_timeseries)} files")
                print(f"    Videos available: V={has_voltage_video}, C={has_calcium_video}")
                
                complete_trials[trial_string] = {
                    'voltage_events': voltage_events[0],
                    'calcium_events': calcium_events[0],
                    'voltage_timeseries': voltage_timeseries,
                    'calcium_timeseries': calcium_timeseries,
                    'videos': segment_videos,
                    'trial_row': trial_row
                }
            else:
                print(f"  ✗ INCOMPLETE - Missing files")
        
        return complete_trials
    
    def get_best_video_for_segment(self, videos_dict, data_type, segment_name):
        """Get the best available video for a segment"""
        video_info = videos_dict[data_type]
        
        # Try segment-specific video first
        segment_video = video_info[segment_name]
        print(f"    DEBUG: Looking for segment video: {segment_video}")
        print(f"    DEBUG: Segment video exists: {segment_video.exists()}")
        
        if segment_video.exists():
            print(f"    DEBUG: Using segment video: {segment_video}")
            return segment_video, True  # True = is segment video
            
        # Fall back to full video
        full_video = video_info['full']
        print(f"    DEBUG: Falling back to full video: {full_video}")
        print(f"    DEBUG: Full video exists: {full_video.exists()}")
        
        if full_video.exists():
            print(f"    DEBUG: Using full video: {full_video}")
            return full_video, False  # False = is full video
            
        print(f"    DEBUG: No video found for {data_type} {segment_name}")
        return None, False
    
    def get_best_timeseries_for_segment(self, timeseries_files, segment_name):
        """Get the best timeseries file for a segment"""
        # Try to find segment-specific file first
        for ts_file in timeseries_files:
            if segment_name in ts_file.name.lower():
                return ts_file
        
        # Fall back to any available file (will filter by segment column later)
        return timeseries_files[0] if timeseries_files else None
    
    def load_segment_data(self, trial_string, trial_files, segment_name, data_type):
        """Load data for a specific segment and data type using segment column filtering"""
        print(f"\nLoading {segment_name} {data_type} data for trial: {trial_string}")
        
        # Load event data and filter by segment
        if data_type == 'voltage':
            events_df = pd.read_csv(trial_files['voltage_events'])
            timeseries_files = trial_files['voltage_timeseries']
        else:
            events_df = pd.read_csv(trial_files['calcium_events'])
            timeseries_files = trial_files['calcium_timeseries']
        
        # Filter events by segment using segment column
        if 'segment' in events_df.columns:
            segment_events = events_df[events_df['segment'] == segment_name].reset_index(drop=True)
            print(f"  Events in {segment_name} segment: {len(segment_events)} (from {len(events_df)} total)")
        else:
            print(f"  Warning: No 'segment' column found, using all events")
            segment_events = events_df
        
        # Get best video for this segment
        video_file, is_segment_video = self.get_best_video_for_segment(
            trial_files['videos'], data_type, segment_name
        )
        
        if video_file is None:
            print(f"No video file found for {data_type}")
            return None
            
        print(f"  Video: {video_file} ({'segment' if is_segment_video else 'full'})")
        
        # Load video frames
        cap = cv2.VideoCapture(str(video_file))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        print(f"  Loaded {len(frames)} video frames")
        
        # Get best timeseries file for this segment
        timeseries_file = self.get_best_timeseries_for_segment(timeseries_files, segment_name)
        if timeseries_file is None:
            print(f"No timeseries file found for {segment_name}")
            return None
            
        timeseries_df = pd.read_csv(timeseries_file)
        print(f"  Timeseries: {timeseries_file.name} ({timeseries_df.shape})")
        
        # Extract cell positions and timeseries data
        cell_positions = timeseries_df[['cell_x', 'cell_y']].copy()
        
        # Find timeseries data start
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
            'timeseries_data': timeseries_data,
            'cell_positions': cell_positions,
            'frames': frames,
            'trial_string': trial_string,
            'segment_name': segment_name,
            'data_type': data_type,
            'sampling_rate': 5,
            'time_axis': np.arange(timeseries_data.shape[1]) / 5,
            'is_segment_video': is_segment_video
        }
    
    def create_interactive_interface(self, segment_data):
        """Create interface for segment-specific data"""
        self.current_trial_data = segment_data
        self.current_frame = 0
        self.selected_events = set()
        self.selected_cells = set()
        
        # Create figure
        self.fig = plt.figure(figsize=(20, 12))
        
        # Left: Timeseries (full left half)
        self.ax_timeseries = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
        
        # Right top: Video
        self.ax_video = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
        
        # Right bottom: Controls
        self.ax_controls = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
        
        self.setup_controls()
        self.setup_event_handlers()
        
        # Auto-play once on startup
        self.start_auto_play()
        
        self.update_display()
        return self.fig
    
    def setup_controls(self):
        """Setup interactive controls"""
        self.ax_controls.axis('off')
        
        # Button layout (using figure coordinates)
        button_height = 0.04
        button_width = 0.12
        base_x = 0.52
        base_y = 0.02
        col1_x = base_x
        col2_x = base_x + button_width + 0.02
        
        row_positions = [
            base_y, base_y + 0.06, base_y + 0.12, base_y + 0.18,
            base_y + 0.24, base_y + 0.30, base_y + 0.36
        ]
        
        # Auto play button
        ax_auto_play = plt.axes([col1_x, row_positions[6], button_width*2 + 0.02, button_height])
        self.btn_auto_play = Button(ax_auto_play, 'Auto Play Once')
        self.btn_auto_play.on_clicked(self.start_auto_play)
        
        # Zoom controls
        ax_zoom_in = plt.axes([col1_x, row_positions[5], button_width, button_height])
        self.btn_zoom_in = Button(ax_zoom_in, 'Zoom+')
        self.btn_zoom_in.on_clicked(lambda x: self.change_zoom(1.5))
        
        ax_zoom_out = plt.axes([col2_x, row_positions[5], button_width, button_height])
        self.btn_zoom_out = Button(ax_zoom_out, 'Zoom-')
        self.btn_zoom_out.on_clicked(lambda x: self.change_zoom(0.75))
        
        # Reset zoom
        ax_reset_zoom = plt.axes([col1_x, row_positions[4], button_width*2 + 0.02, button_height])
        self.btn_reset_zoom = Button(ax_reset_zoom, 'Reset Zoom')
        self.btn_reset_zoom.on_clicked(self.reset_zoom)
        
        # QC Decision buttons
        ax_accept = plt.axes([col1_x, row_positions[3], button_width, button_height])
        self.btn_accept = Button(ax_accept, 'Accept All')
        self.btn_accept.on_clicked(self.accept_trial)
        
        ax_reject_selected = plt.axes([col2_x, row_positions[3], button_width, button_height])
        self.btn_reject_selected = Button(ax_reject_selected, 'Reject Selected')
        self.btn_reject_selected.on_clicked(self.reject_selected)
        
        ax_accept_remaining = plt.axes([col1_x, row_positions[2], button_width, button_height])
        self.btn_accept_remaining = Button(ax_accept_remaining, 'Accept Remaining')
        self.btn_accept_remaining.on_clicked(self.accept_remaining)
        
        ax_reject_all = plt.axes([col2_x, row_positions[2], button_width, button_height])
        self.btn_reject_all = Button(ax_reject_all, 'Reject All')
        self.btn_reject_all.on_clicked(self.reject_trial)
        
        # Next trial button
        ax_next_trial = plt.axes([col1_x, row_positions[1], button_width*2 + 0.02, button_height])
        self.btn_next = Button(ax_next_trial, 'Next Trial')
        self.btn_next.on_clicked(self.next_trial)
        
        # Info display
        self.info_text = self.fig.text(0.75, 0.25, '', fontsize=9, verticalalignment='top',
                                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    def setup_event_handlers(self):
        """Setup mouse and keyboard event handlers"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes == self.ax_video:
            self.handle_video_click(event)
        elif event.inaxes == self.ax_timeseries:
            self.handle_timeseries_click(event)
    
    def handle_video_click(self, event):
        """Handle clicks on video - select cells"""
        if event.xdata is None or event.ydata is None:
            return
            
        click_x, click_y = event.xdata, event.ydata
        
        # Find nearest cell
        cell_positions = self.current_trial_data['cell_positions']
        distances = np.sqrt((cell_positions['cell_x'] - click_x)**2 + 
                           (cell_positions['cell_y'] - click_y)**2)
        nearest_cell = distances.idxmin()
        
        # Toggle cell selection
        if nearest_cell in self.selected_cells:
            self.selected_cells.remove(nearest_cell)
        else:
            self.selected_cells.add(nearest_cell)
            
        # Update selected events based on selected cells
        self.update_selected_events_from_cells()
        self.update_display()
        
    def handle_timeseries_click(self, event):
        """Handle clicks on timeseries - select events and play them"""
        if event.xdata is None or event.ydata is None:
            return
        
        click_time = event.xdata
        click_y = event.ydata
        
        # Find events near click
        events_df = self.current_trial_data['events_df']
        tolerance = 10.0
        
        nearby_events = events_df[
            (events_df['start_time_sec'] <= click_time + tolerance) &
            (events_df['end_time_sec'] >= click_time - tolerance)
        ]
        
        if len(nearby_events) > 0:
            # Find closest event
            distances = []
            for idx, event in nearby_events.iterrows():
                time_dist = abs((event['start_time_sec'] + event['end_time_sec'])/2 - click_time)
                cell_y_pos = event['cell_index'] * 2.0  # Channel spacing
                y_dist = abs(cell_y_pos - click_y)
                distances.append(time_dist + y_dist * 0.1)
            
            closest_idx = nearby_events.index[np.argmin(distances)]
            event_data = nearby_events.loc[closest_idx]
            
            # Toggle event selection
            if closest_idx in self.selected_events:
                self.selected_events.remove(closest_idx)
                print(f"Deselected event {closest_idx}")
            else:
                self.selected_events.add(closest_idx)
                print(f"Selected event {closest_idx}")
            
            # Play the event
            self.play_event(event_data)
            self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == ' ':  # Spacebar
            self.start_auto_play()
        elif event.key == 'left':
            self.current_frame = max(0, self.current_frame - 10)
            self.update_video_display()
        elif event.key == 'right':
            max_frame = len(self.current_trial_data['frames']) - 1
            self.current_frame = min(max_frame, self.current_frame + 10)
            self.update_video_display()
    
    def start_auto_play(self, event=None):
        """Start auto-playing the video"""
        if self.auto_play_active:
            self.auto_play_active = False
            return
            
        self.auto_play_active = True
        self.current_frame = 0
        threading.Thread(target=self.auto_play_loop, daemon=True).start()
    
    def auto_play_loop(self):
        """Auto-play loop"""
        max_frame = len(self.current_trial_data['frames']) - 1
        
        while self.auto_play_active and self.current_frame < max_frame:
            self.current_frame += 1
            
            try:
                self.update_video_display()
                self.fig.canvas.draw_idle()
            except:
                break
            
            time.sleep(0.1)  # 10 FPS
            
        self.auto_play_active = False
    
    def play_event(self, event_data):
        """Play video for a specific event"""
        if self.event_play_active:
            return
            
        start_time = max(0, event_data['start_time_sec'] - 2.0)
        end_time = min(self.current_trial_data['time_axis'][-1], 
                       event_data['end_time_sec'] + 2.0)
        
        start_frame = int(start_time * self.current_trial_data['sampling_rate'])
        end_frame = int(end_time * self.current_trial_data['sampling_rate'])
        
        print(f"Playing event from {start_time:.1f}s to {end_time:.1f}s")
        
        self.event_play_active = True
        self.current_frame = start_frame
        
        threading.Thread(target=self.event_play_loop, args=(start_frame, end_frame), daemon=True).start()
    
    def event_play_loop(self, start_frame, end_frame):
        """Event play loop"""
        current = start_frame
        max_frame = min(end_frame, len(self.current_trial_data['frames']) - 1)
        
        while self.event_play_active and current <= max_frame:
            self.current_frame = current
            
            try:
                self.update_video_display()
                self.fig.canvas.draw_idle()
            except:
                break
            
            current += 1
            time.sleep(0.15)
            
        self.event_play_active = False
    
    def change_zoom(self, factor):
        """Change zoom level"""
        self.zoom_factor *= factor
        self.zoom_factor = max(0.5, min(5.0, self.zoom_factor))
        self.update_video_display()
        
    def reset_zoom(self, event):
        """Reset zoom"""
        self.zoom_factor = 1.0
        self.zoom_center = (256, 256)
        self.update_video_display()
    
    def update_selected_events_from_cells(self):
        """Update selected events based on selected cells"""
        events_df = self.current_trial_data['events_df']
        cell_events = events_df[events_df['cell_index'].isin(self.selected_cells)]
        self.selected_events.update(cell_events.index)
    
    def update_display(self):
        """Update all displays"""
        self.update_video_display()
        self.update_timeseries_display()
        self.update_info_display()
        self.fig.canvas.draw_idle()
    
    def update_video_display(self):
        """Update video display"""
        self.ax_video.clear()
        
        frame = self.current_trial_data['frames'][self.current_frame]
        
        # Apply zoom if needed
        if self.zoom_factor > 1:
            h, w = frame.shape
            crop_h = int(h / self.zoom_factor)
            crop_w = int(w / self.zoom_factor)
            
            center_y, center_x = self.zoom_center
            start_y = max(0, int(center_y - crop_h/2))
            end_y = min(h, start_y + crop_h)
            start_x = max(0, int(center_x - crop_w/2))
            end_x = min(w, start_x + crop_w)
            
            frame = frame[start_y:end_y, start_x:end_x]
        
        self.ax_video.imshow(frame, cmap='gray')
        
        # Overlay cell positions
        cell_positions = self.current_trial_data['cell_positions']
        current_time = self.current_frame / self.current_trial_data['sampling_rate']
        
        # Find active events at current time
        events_df = self.current_trial_data['events_df']
        active_events = events_df[
            (events_df['start_time_sec'] <= current_time) &
            (events_df['end_time_sec'] >= current_time)
        ]
        active_cells = set(active_events['cell_index'])
        
        for idx, (x, y) in cell_positions.iterrows():
            # Adjust for zoom
            if self.zoom_factor > 1:
                x = (x - start_x) * self.zoom_factor
                y = (y - start_y) * self.zoom_factor
            
            # Color coding
            if idx in self.selected_cells:
                color = 'red'
                linewidth = 3
            elif idx in active_cells:
                color = 'orange'
                linewidth = 2
            else:
                color = 'white'
                linewidth = 1
                
            circle = Circle((x, y), radius=8/self.zoom_factor, 
                          fill=False, color=color, linewidth=linewidth)
            self.ax_video.add_patch(circle)
            
            self.ax_video.text(x, y, str(idx), color='yellow', fontsize=8,
                             ha='center', va='center', weight='bold')
        
        current_time = self.current_frame / self.current_trial_data['sampling_rate']
        self.ax_video.set_title(f'Frame {self.current_frame} ({current_time:.1f}s) - Zoom: {self.zoom_factor:.1f}x')
        self.ax_video.axis('off')
    
    def update_timeseries_display(self):
        """Update timeseries display with event bars"""
        self.ax_timeseries.clear()
        
        timeseries_data = self.current_trial_data['timeseries_data']
        time_axis = self.current_trial_data['time_axis']
        events_df = self.current_trial_data['events_df']

        n_channels = timeseries_data.shape[0]
        channel_spacing = 2.0
        
        for i in range(n_channels):
            offset = i * channel_spacing
            
            # Highlight selected cells
            color = 'red' if i in self.selected_cells else 'black'
            alpha = 1.0 if i in self.selected_cells else 0.7
            linewidth = 1.0 if i in self.selected_cells else 0.5
            
            # Normalize and plot timeseries
            ts_data = timeseries_data.iloc[i]
            ts_normalized = (ts_data - ts_data.mean()) / ts_data.std() * 0.8
            
            self.ax_timeseries.plot(time_axis, ts_normalized + offset, 
                                  color=color, linewidth=linewidth, alpha=alpha)
            
            # Add event bars
            channel_events = events_df[events_df['cell_index'] == i]
            for idx, event in channel_events.iterrows():
                start_time = event['start_time_sec']
                end_time = event['end_time_sec']
                
                if idx in self.selected_events:
                    bar_color = 'red'
                    bar_alpha = 0.9
                elif event['event_type'] == 'positive':
                    bar_color = 'blue'
                    bar_alpha = 0.7
                else:
                    bar_color = 'purple'
                    bar_alpha = 0.7
                
                event_rect = Rectangle((start_time, offset - 0.75), 
                                    end_time - start_time, 1.5,
                                    facecolor=bar_color, alpha=bar_alpha)
                self.ax_timeseries.add_patch(event_rect)
            
            # Channel label
            self.ax_timeseries.text(-time_axis[-1]*0.02, offset, f'{i}', 
                                  fontsize=8, va='center')
        
        # Current time indicator
        current_time = self.current_frame / self.current_trial_data['sampling_rate']
        self.ax_timeseries.axvline(current_time, color='green', linewidth=2, alpha=0.8)
        
        self.ax_timeseries.set_xlabel('Time (s)')
        self.ax_timeseries.set_ylabel('Channel')
        self.ax_timeseries.set_xlim(0, time_axis[-1])
        self.ax_timeseries.set_ylim(-1, n_channels * channel_spacing)
        self.ax_timeseries.grid(True, alpha=0.3)
    
    def update_info_display(self):
        """Update info text with segment information"""
        events_df = self.current_trial_data['events_df']
        total_events = len(events_df)
        selected_events = len(self.selected_events)
        selected_cells = len(self.selected_cells)
        
        info_text = f"""Trial: {self.current_trial_data['trial_string']}
                Segment: {self.current_trial_data['segment_name'].upper()}
                Data: {self.current_trial_data['data_type'].title()}
                Video: {'Segment-specific' if self.current_trial_data['is_segment_video'] else 'Full recording'}

                Total Events: {total_events}
                Selected Events: {selected_events}
                Selected Cells: {selected_cells}

                Frame: {self.current_frame}/{len(self.current_trial_data['frames'])-1}
                Time: {self.current_frame/self.current_trial_data['sampling_rate']:.1f}s

                Click cells to select
                Click events to select/play"""
        
        self.info_text.set_text(info_text)
    
    # QC decision methods - updated for segments
    def accept_trial(self, event):
        """Accept all events in this segment"""
        trial_string = self.current_trial_data['trial_string']
        segment_name = self.current_trial_data['segment_name']
        data_type = self.current_trial_data['data_type']
        
        qc_key = f"{trial_string}_{segment_name}_{data_type}"
        
        self.qc_decisions[qc_key] = {
            'trial_string': trial_string,
            'segment': segment_name,
            'data_type': data_type,
            'decision': 'accept',
            'rejected_events': []
        }
        
        print(f"Accepted all events for {trial_string} {segment_name} {data_type}")
        plt.close(self.fig)
    
    def reject_selected(self, event):
        """Reject selected events in this segment"""
        if not self.selected_events and not self.selected_cells:
            print("No events selected.")
            return
        
        trial_string = self.current_trial_data['trial_string']
        segment_name = self.current_trial_data['segment_name']
        data_type = self.current_trial_data['data_type']
        
        qc_key = f"{trial_string}_{segment_name}_{data_type}"
        
        events_to_reject = set(self.selected_events)
        
        if self.selected_cells:
            events_df = self.current_trial_data['events_df']
            cell_events = events_df[events_df['cell_index'].isin(self.selected_cells)]
            events_to_reject.update(cell_events.index)
        
        self.qc_decisions[qc_key] = {
            'trial_string': trial_string,
            'segment': segment_name,
            'data_type': data_type,
            'decision': 'partial_reject',
            'rejected_events': list(events_to_reject)
        }
        
        print(f"Rejected {len(events_to_reject)} events for {trial_string} {segment_name} {data_type}")
        plt.close(self.fig)
    
    def accept_remaining(self, event):
        """Accept remaining events in this segment"""
        self.reject_selected(event)
    
    def reject_trial(self, event):
        """Reject all events in this segment"""
        trial_string = self.current_trial_data['trial_string']
        segment_name = self.current_trial_data['segment_name']
        data_type = self.current_trial_data['data_type']
        
        qc_key = f"{trial_string}_{segment_name}_{data_type}"
        
        self.qc_decisions[qc_key] = {
            'trial_string': trial_string,
            'segment': segment_name,
            'data_type': data_type,
            'decision': 'reject',
            'rejected_events': []
        }
        
        print(f"Rejected all events for {trial_string} {segment_name} {data_type}")
        plt.close(self.fig)
    
    def next_trial(self, event):
        """Skip this segment"""
        print("Skipping segment...")
        plt.close(self.fig)
    
    def save_qc_results(self):
        """Save QC decisions"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.base_results_dir / f"qc_decisions_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.qc_decisions, f, indent=2, default=str)
        
        print(f"QC decisions saved to: {results_file}")
    
    def apply_qc_decisions_segments(self):
        """Apply segment-based QC decisions to create final filtered event files"""
        if not self.qc_decisions:
            print("No QC decisions to apply.")
            return
        
        print("Applying segment-based QC decisions...")
        
        # Group decisions by trial and data type
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
        
        # Apply decisions for each trial
        for trial_string, trial_data in trial_decisions.items():
            for data_type, segment_decisions in trial_data.items():
                
                trial_dir = self.base_results_dir / trial_string
                
                # Find source event file
                if data_type == 'voltage':
                    source_files = list(trial_dir.glob("events_voltage_*_filtered.csv"))
                else:
                    source_files = list(trial_dir.glob("events_calcium_*_filtered.csv"))
                
                if not source_files:
                    print(f"  ✗ No source file found for {trial_string} {data_type}")
                    continue
                    
                source_file = source_files[0]
                events_df = pd.read_csv(source_file)
                
                # Apply QC decisions segment by segment
                final_events_list = []
                
                for segment in ['pre', 'post']:
                    if segment in segment_decisions:
                        decision_info = segment_decisions[segment]
                        decision = decision_info['decision']
                        rejected_events = decision_info.get('rejected_events', [])
                        
                        # Filter events for this segment
                        if 'segment' in events_df.columns:
                            segment_events = events_df[events_df['segment'] == segment]
                        else:
                            # If no segment column, we can't separate - use all events
                            segment_events = events_df
                        
                        if decision == 'accept':
                            final_segment_events = segment_events
                            print(f"  ✓ Accepted: {trial_string} {data_type} {segment} ({len(final_segment_events)} events)")
                            
                        elif decision == 'partial_reject':
                            final_segment_events = segment_events[~segment_events.index.isin(rejected_events)]
                            print(f"  ◐ Partial: {trial_string} {data_type} {segment} "
                                  f"({len(final_segment_events)}/{len(segment_events)} events kept)")
                            
                        elif decision == 'reject':
                            final_segment_events = segment_events.iloc[0:0].copy()
                            print(f"  ✗ Rejected: {trial_string} {data_type} {segment} (0 events)")
                        
                        final_events_list.append(final_segment_events)
                    
                    else:
                        # No QC decision for this segment - keep all events
                        if 'segment' in events_df.columns:
                            segment_events = events_df[events_df['segment'] == segment]
                        else:
                            segment_events = events_df
                        final_events_list.append(segment_events)
                        print(f"  ? No QC: {trial_string} {data_type} {segment} ({len(segment_events)} events kept)")
                
                # Combine all segments
                if final_events_list:
                    final_events = pd.concat(final_events_list, ignore_index=True)
                else:
                    final_events = events_df.iloc[0:0].copy()  # Empty DataFrame
                
                # Save final QC'd events
                final_file = trial_dir / f"events_{data_type}_{trial_string}_segment_QC_final.csv"
                final_events.to_csv(final_file, index=False)
                
                print(f"  📁 Final file: {final_file.name} ({len(final_events)} total events)")
        
        print("Segment-based QC application complete!")
        print("Final files saved as: events_{type}_{trial}_segment_QC_final.csv")
    
    def run_qc_session(self):
        """Main QC session - uses segment column to filter events dynamically"""
        print("Starting Segment-Based Interactive Event QC Session")
        print("="*60)
        print("Using existing pipeline structure with segment column filtering")
        print("Each segment (pre/post) reviewed separately for each data type")
        print("Controls:")
        print("  - Click cells in video to select/deselect")
        print("  - Click colored event bars in timeseries to select/play events")
        print("  - Auto-play shows recording (segment or full)")
        print("  - Space=auto-play, Left/Right arrows=jump frames")
        print("="*60)
        
        # Find trials
        trials = self.find_trials_from_metadata()
        
        if not trials:
            print("No complete trial datasets found!")
            return
        
        # Simple calculation - no nested structure needed
        num_trials = len(trials)
        segments_per_trial = 2  # pre and post
        data_types_per_segment = 2  # voltage and calcium
        total_reviews = num_trials * segments_per_trial * data_types_per_segment
        
        print(f"Found {num_trials} complete trials")
        print(f"Will review {total_reviews} total segments (pre/post × voltage/calcium)")
        
        # Process each combination
        review_count = 0
        for trial_string, trial_files in trials.items():
            for segment_name in ['pre', 'post']:
                for data_type in ['voltage', 'calcium']:
                    review_count += 1
                    
                    print(f"\n{'='*60}")
                    print(f"QC {review_count}/{total_reviews}: {trial_string} - {segment_name.upper()} - {data_type.upper()}")
                    print(f"{'='*60}")
                    
                    # Load segment data
                    try:
                        segment_trial_data = self.load_segment_data(
                            trial_string, trial_files, segment_name, data_type
                        )
                        
                        if segment_trial_data is None:
                            print(f"Failed to load data for {trial_string} {segment_name} {data_type}")
                            continue
                        
                        # Check if this segment has any events
                        if len(segment_trial_data['events_df']) == 0:
                            print(f"No events found in {segment_name} {data_type} - auto-accepting")
                            # Auto-accept empty segments
                            qc_key = f"{trial_string}_{segment_name}_{data_type}"
                            self.qc_decisions[qc_key] = {
                                'trial_string': trial_string,
                                'segment': segment_name,
                                'data_type': data_type,
                                'decision': 'accept',
                                'rejected_events': []
                            }
                            continue
                        
                        print(f"Found {len(segment_trial_data['events_df'])} events to review")
                        
                        # Create interactive interface
                        fig = self.create_interactive_interface(segment_trial_data)
                        
                        # Show interface (blocks until user closes window)
                        plt.show(block=True)
                        
                    except Exception as e:
                        print(f"Error processing {trial_string} {segment_name} {data_type}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        # Save and apply results
        self.save_qc_results()
        
        apply_now = input("\nApply QC decisions now? (y/n): ").strip().lower()
        if apply_now in ['y', 'yes']:
            self.apply_qc_decisions_segments()


def main():
    """Main function - integrated with pipeline structure"""
    print("Updated Interactive Event QC Tool")
    print("Integrated with new pipeline structure")
    
    # Setup paths (same as your create_paper_data.py)
    home = Path.home()
    cell_line = 'MDA_MB_468'
    date = '20250827'
    if "ys5320" in str(home):
        top_dir = Path(home, "firefly_link/Calcium_Voltage_Imaging", f'{cell_line}')
        df_file = Path(top_dir, 'analysis', 'dataframes', f'long_acqs_{cell_line}_all_before_{date}.csv')
    else:
        # Add local configuration
        top_dir = Path(r"R:\home\firefly_link\Calcium_Voltage_Imaging", f'{cell_line}')
        df_file = Path(top_dir, 'analysis', 'dataframes', f'long_acqs_{cell_line}_all_before_{date}.csv')
    
    # Load and filter metadata (same as your create_paper_data.py)
    print(f"Loading metadata from: {df_file}")
    
    if not df_file.exists():
        print(f"Metadata file not found: {df_file}")
        return
    
    df_raw = pd.read_csv(df_file)
    
    # Apply same filtering as your pipeline
    df_filtered = df_raw[df_raw['multi_tif'] > 1]
    df_filtered = df_filtered[df_filtered['use'] == 'y']
    
    # Optional: Filter by specific experiment (uncomment to use)
    df_filtered = df_filtered[df_filtered['expt'] == 'TRAM-34_1uM']
    
    df_filtered = df_filtered.reset_index(drop=True)
    
    print(f"After filtering: {len(df_filtered)} trials available for QC")
    
    if len(df_filtered) == 0:
        print("No trials passed filtering. Check your filtering criteria.")
        return
    
    # Show available trials
    print("\nAvailable trials:")
    for idx, row in df_filtered.iterrows():
        print(f"  {idx}: {row.trial_string}")
    
    # Optional: Allow user to select subset
    user_input = input(f"\nQC all {len(df_filtered)} trials? (y/n): ").strip().lower()
    if user_input not in ['y', 'yes']:
        print("You can modify the filtering in the script to select specific trials.")
        return
    
    # Create QC tool
    qc_tool = UpdatedInteractiveEventQC(df_filtered, top_dir)
    
    # Run QC session
    qc_tool.run_qc_session()


if __name__ == "__main__":
    main()