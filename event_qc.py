"""
Advanced Interactive Event Quality Control Tool
Features: Real-time video playback, click-to-select events, zoom controls
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

class AdvancedInteractiveEventQC:
    def __init__(self, base_results_dir, data_dir, imaging_base_dir):
        self.base_results_dir = Path(base_results_dir)
        self.data_dir = Path(data_dir)
        self.imaging_base_dir = Path(imaging_base_dir)
        self.qc_decisions = {}
        
        # Video playback state
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self.zoom_factor = 1.0
        self.zoom_center = (256, 256)  # Center of 512x512 image
        
        # Event selection state
        self.selected_events = set()
        self.selected_cells = set()
        self.selected_time_ranges = []
        
        # UI elements
        self.fig = None
        self.ax_video = None
        self.ax_timeseries = None
        self.ax_timeline = None
        self.current_trial_data = None
    def find_imaging_folder_for_trial(self, trial_string):
        """Find video folder instead of TIFF folder"""
        #video_base_dir = Path(r'R:\home\firefly_link\Calcium_Voltage_Imaging\MDA_MB_468\analysis\results_profiles')
        video_base_dir = Path(self.imaging_base_dir, 'analysis','results_profiles')
        trial_video_dir = video_base_dir / trial_string
        
        if trial_video_dir.exists():
            # Check if both videos exist
            voltage_video = trial_video_dir / "enhanced_voltage_video.avi"
            calcium_video = trial_video_dir / "enhanced_calcium_video.avi"
            
            if voltage_video.exists() and calcium_video.exists():
                print(f"  Found video folder: {trial_video_dir}")
                return trial_video_dir
        
        print(f"  No video folder found for: {trial_string}")
        return None
        
    def find_trial_files(self):
        """Find all trials with complete data sets"""
        trials = {}
        
        for trial_dir in self.base_results_dir.iterdir():
            if not trial_dir.is_dir():
                continue
                
            trial_string = trial_dir.name
            print(f"Checking trial: {trial_string}")

            # Check what files exist
            voltage_events = list(trial_dir.glob("events_voltage_*_filtered.csv"))
            calcium_events = list(trial_dir.glob("events_calcium_*_filtered.csv"))
            voltage_data = list(self.data_dir.glob(f"*{trial_string}*_voltage_transfected*.csv"))
            calcium_data = list(self.data_dir.glob(f"*{trial_string}*_ca_transfected*.csv"))

            print(f"  Voltage events found: {len(voltage_events)}")
            print(f"  Calcium events found: {len(calcium_events)}")
            print(f"  Voltage data found: {len(voltage_data)}")
            print(f"  Calcium data found: {len(calcium_data)}")

            if voltage_events: print(f"    Example voltage event: {voltage_events[0].name}")
            if voltage_data: print(f"    Example voltage data: {voltage_data[0].name}")
            
            voltage_events = list(trial_dir.glob("events_voltage_*_filtered.csv"))
            calcium_events = list(trial_dir.glob("events_calcium_*_filtered.csv"))
            voltage_data = list(self.data_dir.glob(f"*{trial_string}*_voltage_transfected*.csv"))
            calcium_data = list(self.data_dir.glob(f"*{trial_string}*_ca_transfected*.csv"))
            imaging_folder = self.find_imaging_folder_for_trial(trial_string)
            
            if (voltage_events and calcium_events and 
                voltage_data and calcium_data and imaging_folder):
                
                trials[trial_string] = {
                    'voltage_events': voltage_events[0],
                    'calcium_events': calcium_events[0],
                    'voltage_data': voltage_data[0],
                    'calcium_data': calcium_data[0],
                    'imaging_folder': imaging_folder
                }
        
        return trials
    
    def load_trial_data(self, trial_string, trial_files, data_type):
        """Load all data for a specific trial and data type"""
        print(f"\nLoading {data_type} data for trial: {trial_string}")
        
        if data_type == 'voltage':
            events_df = pd.read_csv(trial_files['voltage_events'])
            timeseries_df = pd.read_csv(trial_files['voltage_data'])
        else:
            events_df = pd.read_csv(trial_files['calcium_events'])
            timeseries_df = pd.read_csv(trial_files['calcium_data'])
        
        cell_positions = timeseries_df[['cell_x', 'cell_y']].copy()
        
        # Load pre-made videos instead of raw TIFF files
        video_base_dir = Path(self.imaging_base_dir, 'analysis','results_profiles')
        trial_video_dir = video_base_dir / trial_string

        if data_type == 'voltage':
            video_file = trial_video_dir / "enhanced_voltage_video.avi"
        else:
            video_file = trial_video_dir / "enhanced_calcium_video.avi"

        if not video_file.exists():
            print(f"Video file not found: {video_file}")
            return None

        print(f"Loading video: {video_file}")

        # Load video frames using OpenCV
        cap = cv2.VideoCapture(str(video_file))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        cap.release()
        frames = np.array(frames)
        print(f"Loaded {len(frames)} frames from video")
        
        timeseries_data = timeseries_df.iloc[:, -5000:]
        
        return {
            'events_df': events_df,
            'timeseries_data': timeseries_data,
            'cell_positions': cell_positions,
            'frames': frames,
            'trial_string': trial_string,
            'data_type': data_type,
            'sampling_rate': 5,
            'time_axis': np.arange(timeseries_data.shape[1]) / 5
        }
    
    def normalize_frames(self, frames):
        """Normalize frames for consistent display"""
        normalized = np.zeros_like(frames)
        for i in range(len(frames)):
            frame = frames[i].astype(float)
            frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
            normalized[i] = frame.astype(np.uint8)
        return normalized
    
    def create_interactive_interface(self, trial_data):
        """Create the main interactive interface"""
        self.current_trial_data = trial_data
        self.current_frame = 0
        self.selected_events = set()
        self.selected_cells = set()
        self.selected_time_ranges = []
        
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(20, 12))
        gs = self.fig.add_gridspec(3, 4, height_ratios=[4, 0.3, 3], hspace=0.2, wspace=0.3)
        
        # Video display (top, larger)
        self.ax_video = self.fig.add_subplot(gs[0, :3])  # Takes 3/4 of top width

        # Control panel (top right, smaller)  
        self.ax_controls = self.fig.add_subplot(gs[0, 3:])

        # Timeseries plot (bottom, full width, taller)
        self.ax_timeseries = self.fig.add_subplot(gs[2, :])

        # Frame slider (middle, smaller)
        self.ax_slider = self.fig.add_subplot(gs[1, :3])

        # Remove the timeline subplot completely
        # self.ax_timeline = self.fig.add_subplot(gs[3, :])  # Comment this out
        
        # Frame slider (below video)
        self.ax_slider = self.fig.add_subplot(gs[1, :2])
        
        self.setup_controls()
        self.setup_event_handlers()
        self.update_display()
        
        return self.fig
    
    def setup_controls(self):
        """Setup interactive controls"""
        # Frame slider
        max_frames = len(self.current_trial_data['frames']) - 1
        self.frame_slider = Slider(
            self.ax_slider, 'Frame', 0, max_frames, 
            valinit=0, valfmt='%d', valstep=1
        )
        self.frame_slider.on_changed(self.on_frame_change)
        
        # Control buttons
        button_height = 0.04
        button_width = 0.12
        button_spacing = 0.02
        
        # Play/Pause button
        ax_play = plt.axes([0.52, 0.85, button_width, button_height])
        self.btn_play = Button(ax_play, 'Play/Pause')
        self.btn_play.on_clicked(self.toggle_playback)
        
        # Speed controls
        ax_speed_up = plt.axes([0.52, 0.76, button_width/2, button_height])
        self.btn_speed_up = Button(ax_speed_up, 'Speed+')
        self.btn_speed_up.on_clicked(lambda x: self.change_speed(1.5))
        
        ax_speed_down = plt.axes([0.52 + button_width/2, 0.76, button_width/2, button_height])
        self.btn_speed_down = Button(ax_speed_down, 'Speed-')
        self.btn_speed_down.on_clicked(lambda x: self.change_speed(0.75))
        
        # Zoom controls
        ax_zoom_in = plt.axes([0.52, 0.67, button_width/2, button_height])
        self.btn_zoom_in = Button(ax_zoom_in, 'Zoom+')
        self.btn_zoom_in.on_clicked(lambda x: self.change_zoom(1.5))
        
        ax_zoom_out = plt.axes([0.52 + button_width/2, 0.67, button_width/2, button_height])
        self.btn_zoom_out = Button(ax_zoom_out, 'Zoom-')
        self.btn_zoom_out.on_clicked(lambda x: self.change_zoom(0.75))
        
        # Reset zoom
        ax_reset_zoom = plt.axes([0.52, 0.58, button_width, button_height])
        self.btn_reset_zoom = Button(ax_reset_zoom, 'Reset Zoom')
        self.btn_reset_zoom.on_clicked(self.reset_zoom)
        
        # QC Decision buttons
        ax_accept = plt.axes([0.52, 0.45, button_width, button_height])
        self.btn_accept = Button(ax_accept, 'Accept All')
        self.btn_accept.on_clicked(self.accept_trial)
        
        ax_reject_selected = plt.axes([0.52, 0.36, button_width, button_height])
        self.btn_reject_selected = Button(ax_reject_selected, 'Reject Selected')
        self.btn_reject_selected.on_clicked(self.reject_selected)
        
        ax_accept_remaining = plt.axes([0.52, 0.27, button_width, button_height])
        self.btn_accept_remaining = Button(ax_accept_remaining, 'Accept Remaining')
        self.btn_accept_remaining.on_clicked(self.accept_remaining)
        
        ax_reject_all = plt.axes([0.52, 0.18, button_width, button_height])
        self.btn_reject_all = Button(ax_reject_all, 'Reject All')
        self.btn_reject_all.on_clicked(self.reject_trial)
        
        ax_next_trial = plt.axes([0.52, 0.09, button_width, button_height])
        self.btn_next = Button(ax_next_trial, 'Next Trial')
        self.btn_next.on_clicked(self.next_trial)
        
        # Info display
        self.info_text = self.ax_controls.text(0.7, 0.8, '', transform=self.ax_controls.transAxes,
                                              fontsize=10, verticalalignment='top')
        
    def setup_event_handlers(self):
        """Setup mouse click handlers"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def on_frame_change(self, val):
        """Handle frame slider change"""
        self.current_frame = int(val)
        self.update_video_display()
        
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes == self.ax_video:
            self.handle_video_click(event)
        elif event.inaxes == self.ax_timeseries:
            self.handle_timeseries_click(event)
        elif event.inaxes == self.ax_timeline:
            self.handle_timeline_click(event)
    
    def handle_video_click(self, event):
        """Handle clicks on video display - select cells"""
        if event.xdata is None or event.ydata is None:
            return
            
        # Convert click coordinates to image coordinates
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
        """Handle clicks on timeseries - select individual events"""
        if event.xdata is None or event.ydata is None:
            return
        
        # Check if clicked on an event rectangle
        if hasattr(event.artist, 'get_gid') and event.artist.get_gid():
            event_id = event.artist.get_gid()
            if event_id.startswith('event_'):
                idx = int(event_id.split('_')[1])
                
                # Toggle event selection
                if idx in self.selected_events:
                    self.selected_events.remove(idx)
                    print(f"Deselected event {idx}")
                else:
                    self.selected_events.add(idx)
                    print(f"Selected event {idx} for rejection")
                
                self.update_display()
                return
            
        click_time = event.xdata
        click_y = event.ydata
        
        # Find events near click
        events_df = self.current_trial_data['events_df']
        tolerance = 5.0  # seconds
        
        nearby_events = events_df[
            (events_df['start_time_sec'] <= click_time + tolerance) &
            (events_df['end_time_sec'] >= click_time - tolerance)
        ]
        
        # Find closest event
        if len(nearby_events) > 0:
            # Calculate distances considering both time and cell position
            distances = []
            for idx, event in nearby_events.iterrows():
                time_dist = abs((event['start_time_sec'] + event['end_time_sec'])/2 - click_time)
                cell_dist = abs(event['cell_index'] - click_y/2.0)  # Approximate cell from y position
                distances.append(time_dist + cell_dist)
            
            closest_idx = nearby_events.index[np.argmin(distances)]
            
            # Toggle event selection
            if closest_idx in self.selected_events:
                self.selected_events.remove(closest_idx)
            else:
                self.selected_events.add(closest_idx)
                
            self.update_display()
    
    def handle_timeline_click(self, event):
        """Handle clicks on timeline - select time ranges"""
        # This would implement click-and-drag selection
        # For now, just add clicked time as a point selection
        if event.xdata is None:
            return
            
        click_time = event.xdata
        # Simple point selection - could be enhanced to range selection
        time_range = (click_time - 2.5, click_time + 2.5)  # 5-second window
        
        # Check if this overlaps with existing selections
        overlapping = False
        for i, (start, end) in enumerate(self.selected_time_ranges):
            if not (time_range[1] < start or time_range[0] > end):
                # Remove overlapping selection
                self.selected_time_ranges.pop(i)
                overlapping = True
                break
        
        if not overlapping:
            self.selected_time_ranges.append(time_range)
        
        self.update_selected_events_from_time_ranges()
        self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == ' ':  # Spacebar
            self.toggle_playback(None)
        elif event.key == 'left':
            self.current_frame = max(0, self.current_frame - 1)
            self.frame_slider.set_val(self.current_frame)
        elif event.key == 'right':
            max_frame = len(self.current_trial_data['frames']) - 1
            self.current_frame = min(max_frame, self.current_frame + 1)
            self.frame_slider.set_val(self.current_frame)
    
    def toggle_playback(self, event):
        """Toggle video playback"""
        self.playing = not self.playing
        if self.playing:
            threading.Thread(target=self.playback_loop, daemon=True).start()
    
    def playback_loop(self):
        """Video playback loop"""
        while self.playing:
            max_frame = len(self.current_trial_data['frames']) - 1
            if self.current_frame >= max_frame:
                self.playing = False
                break
                
            self.current_frame += 1
            
            # Update display in main thread
            self.frame_slider.set_val(self.current_frame)
            self.fig.canvas.draw_idle()
            
            # Control playback speed
            time.sleep(0.2 / self.playback_speed)
    
    def change_speed(self, factor):
        """Change playback speed"""
        self.playback_speed *= factor
        self.playback_speed = max(0.1, min(10.0, self.playback_speed))
        
    def change_zoom(self, factor):
        """Change zoom level"""
        self.zoom_factor *= factor
        self.zoom_factor = max(0.5, min(5.0, self.zoom_factor))
        self.update_video_display()
        
    def reset_zoom(self, event):
        """Reset zoom to original"""
        self.zoom_factor = 1.0
        self.zoom_center = (256, 256)
        self.update_video_display()
    
    def update_selected_events_from_cells(self):
        """Update selected events based on selected cells"""
        events_df = self.current_trial_data['events_df']
        cell_events = events_df[events_df['cell_index'].isin(self.selected_cells)]
        self.selected_events.update(cell_events.index)
    
    def update_selected_events_from_time_ranges(self):
        """Update selected events based on selected time ranges"""
        events_df = self.current_trial_data['events_df']
        
        for start_time, end_time in self.selected_time_ranges:
            range_events = events_df[
                (events_df['start_time_sec'] >= start_time) &
                (events_df['end_time_sec'] <= end_time)
            ]
            self.selected_events.update(range_events.index)
    
    def update_display(self):
        """Update all display elements"""
        self.update_video_display()
        self.update_timeseries_display()
        #self.update_timeline_display()
        self.update_info_display()
        self.fig.canvas.draw_idle()
    
    def update_video_display(self):
        """Update video frame display"""
        self.ax_video.clear()
        
        # Display current frame with zoom
        frame = self.current_trial_data['frames'][self.current_frame]
        
        if self.zoom_factor > 1:
            # Calculate crop region
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
        
        # Overlay cell positions and selections
        cell_positions = self.current_trial_data['cell_positions']
        current_time = self.current_frame / self.current_trial_data['sampling_rate']
        
        # Get events active at current frame
        events_df = self.current_trial_data['events_df']
        active_events = events_df[
            (events_df['start_time_sec'] <= current_time) &
            (events_df['end_time_sec'] >= current_time)
        ]
        active_cells = set(active_events['cell_index'])
        
        for idx, (x, y) in cell_positions.iterrows():
            # Adjust coordinates for zoom
            if self.zoom_factor > 1:
                x = (x - start_x) * self.zoom_factor
                y = (y - start_y) * self.zoom_factor
            
            # Color coding: red=selected, orange=active event, white=normal
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
            
            # Add cell number
            self.ax_video.text(x, y, str(idx), color='yellow', fontsize=8,
                             ha='center', va='center', weight='bold')
        
        self.ax_video.set_title(f'Frame {self.current_frame} ({current_time:.1f}s) - '
                               f'Zoom: {self.zoom_factor:.1f}x - Speed: {self.playback_speed:.1f}x')
        self.ax_video.axis('off')
    
    def update_timeseries_display(self):
        """Update timeseries plot with selections highlighted"""
        self.ax_timeseries.clear()
        
        timeseries_data = self.current_trial_data['timeseries_data']
        time_axis = self.current_trial_data['time_axis']
        events_df = self.current_trial_data['events_df']
        
        # Plot subset of channels
        n_channels_plot = timeseries_data.shape[0]
        channel_spacing = 1.0
        
        for i in range(n_channels_plot):
            offset = i * channel_spacing
            
            # Highlight selected cells
            if i in self.selected_cells:
                color = 'red'
                alpha = 1.0
                linewidth = 1.0
            else:
                color = 'black'
                alpha = 0.7
                linewidth = 0.5
                
            self.ax_timeseries.plot(time_axis, timeseries_data.iloc[i] + offset, 
                                  color=color, linewidth=linewidth, alpha=alpha)
            
            # Highlight events
            # Highlight events with clickable rectangles
            channel_events = events_df[events_df['cell_index'] == i]
            for idx, event in channel_events.iterrows():
                start_time = event['start_time_sec']
                end_time = event['end_time_sec']
                
                if idx in self.selected_events:
                    color = 'red'
                    alpha = 0.9
                    linewidth = 3  # Thicker for selected events
                elif event['event_type'] == 'positive':
                    color = 'blue'
                    alpha = 0.7
                    linewidth = 2
                else:
                    color = 'purple'
                    alpha = 0.7
                    linewidth = 2
                
                # Add clickable event rectangle
                event_rect = Rectangle((start_time, i*channel_spacing - 0.3), 
                                    end_time - start_time, 0.6,
                                    facecolor=color, alpha=alpha, 
                                    picker=True, gid=f'event_{idx}')  # Make pickable
                self.ax_timeseries.add_patch(event_rect)
            
            # Add channel label
            self.ax_timeseries.text(-time_axis[-1]*0.02, offset, f'{i}', 
                                  fontsize=8, va='center')
        
        # Add current time indicator
        current_time = self.current_frame / self.current_trial_data['sampling_rate']
        self.ax_timeseries.axvline(current_time, color='green', linewidth=2, alpha=0.8)
        
        # Highlight selected time ranges
        for start_time, end_time in self.selected_time_ranges:
            self.ax_timeseries.axvspan(start_time, end_time, alpha=0.2, color='yellow')
        
        self.ax_timeseries.set_xlabel('Time (s)')
        self.ax_timeseries.set_ylabel('Channel')
        self.ax_timeseries.set_xlim(0, time_axis[-1])
        self.ax_timeseries.grid(True, alpha=0.3)
    
    def update_timeline_display(self):
        """Update event timeline"""
        self.ax_timeline.clear()
        
        events_df = self.current_trial_data['events_df']
        
        for idx, event in events_df.iterrows():
            if idx in self.selected_events:
                color = 'red'
                alpha = 0.8
            elif event['event_type'] == 'positive':
                color = 'blue'
                alpha = 0.7
            else:
                color = 'purple'
                alpha = 0.7
                
            self.ax_timeline.barh(event['cell_index'], event['duration_sec'], 
                                left=event['start_time_sec'], height=0.8, 
                                color=color, alpha=alpha)
        
        # Current time indicator
        current_time = self.current_frame / self.current_trial_data['sampling_rate']
        self.ax_timeline.axvline(current_time, color='green', linewidth=2, alpha=0.8)
        
        # Selected time ranges
        for start_time, end_time in self.selected_time_ranges:
            rect = Rectangle((start_time, -0.5), end_time - start_time, 
                           len(self.current_trial_data['cell_positions']) + 1,
                           alpha=0.2, color='yellow')
            self.ax_timeline.add_patch(rect)
        
        self.ax_timeline.set_xlabel('Time (s)')
        self.ax_timeline.set_ylabel('Cell Index')
        self.ax_timeline.grid(True, alpha=0.3)
    
    def update_info_display(self):
        """Update information text"""
        events_df = self.current_trial_data['events_df']
        total_events = len(events_df)
        selected_events = len(self.selected_events)
        selected_cells = len(self.selected_cells)
        
        info_text = f"""
Trial: {self.current_trial_data['trial_string']}
Data: {self.current_trial_data['data_type'].title()}

Total Events: {total_events}
Selected Events: {selected_events}
Selected Cells: {selected_cells}
Time Ranges: {len(self.selected_time_ranges)}

Frame: {self.current_frame}/{len(self.current_trial_data['frames'])-1}
Speed: {self.playback_speed:.1f}x
Zoom: {self.zoom_factor:.1f}x
        """
        
        self.info_text.set_text(info_text)
    
    def accept_trial(self, event):
        """Accept all events in trial"""
        trial_string = self.current_trial_data['trial_string']
        data_type = self.current_trial_data['data_type']
        
        if trial_string not in self.qc_decisions:
            self.qc_decisions[trial_string] = {}
        
        self.qc_decisions[trial_string][data_type] = {
            'decision': 'accept',
            'rejected_events': []
        }
        
        print(f"Accepted all events for {trial_string} {data_type}")
        plt.close(self.fig)
    
    def reject_selected(self, event):
        """Reject only selected events"""
        # Continue to detailed selection if nothing selected
        if not self.selected_events and not self.selected_cells and not self.selected_time_ranges:
            print("No events selected. Please select events, cells, or time ranges to reject.")
            return
        
        trial_string = self.current_trial_data['trial_string']
        data_type = self.current_trial_data['data_type']
        
        if trial_string not in self.qc_decisions:
            self.qc_decisions[trial_string] = {}
        
        # Collect all events to reject
        events_to_reject = set(self.selected_events)
        
        # Add events from selected cells
        events_df = self.current_trial_data['events_df']
        if self.selected_cells:
            cell_events = events_df[events_df['cell_index'].isin(self.selected_cells)]
            events_to_reject.update(cell_events.index)
        
        # Add events from selected time ranges
        for start_time, end_time in self.selected_time_ranges:
            range_events = events_df[
                (events_df['start_time_sec'] >= start_time) &
                (events_df['end_time_sec'] <= end_time)
            ]
            events_to_reject.update(range_events.index)
        
        self.qc_decisions[trial_string][data_type] = {
            'decision': 'partial_reject',
            'rejected_events': list(events_to_reject)
        }
        
        print(f"Rejected {len(events_to_reject)} events for {trial_string} {data_type}")
        plt.close(self.fig)
    
    def accept_remaining(self, event):
        """Accept remaining (non-selected) events"""
        self.reject_selected(event)  # Same logic, different framing
    
    def reject_trial(self, event):
        """Reject entire trial"""
        trial_string = self.current_trial_data['trial_string']
        data_type = self.current_trial_data['data_type']
        
        if trial_string not in self.qc_decisions:
            self.qc_decisions[trial_string] = {}
        
        self.qc_decisions[trial_string][data_type] = {
            'decision': 'reject',
            'rejected_events': []
        }
        
        print(f"Rejected all events for {trial_string} {data_type}")
        plt.close(self.fig)
    
    def next_trial(self, event):
        """Skip to next trial"""
        print("Skipping trial...")
        plt.close(self.fig)
    
    def save_qc_results(self):
        """Save QC decisions"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.base_results_dir / f"advanced_qc_decisions_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.qc_decisions, f, indent=2, default=str)
        
        print(f"QC decisions saved to: {results_file}")
    
    def apply_qc_decisions(self):
        """Apply QC decisions to create filtered event files"""
        if not self.qc_decisions:
            print("No QC decisions to apply.")
            return
        
        print("Applying QC decisions...")
        
        for trial_string, decisions in self.qc_decisions.items():
            for data_type, decision_info in decisions.items():
                decision = decision_info['decision']
                rejected_events = decision_info.get('rejected_events', [])
                
                trial_dir = self.base_results_dir / trial_string
                
                if data_type == 'voltage':
                    source_file = list(trial_dir.glob("events_voltage_*_filtered.csv"))[0]
                else:
                    source_file = list(trial_dir.glob("events_calcium_*_filtered.csv"))[0]
                
                events_df = pd.read_csv(source_file)
                
                if decision == 'accept':
                    # Keep all events
                    final_events = events_df
                    print(f"  ✓ Accepted: {trial_string} {data_type} ({len(final_events)} events)")
                    
                elif decision == 'partial_reject':
                    # Remove only rejected events
                    final_events = events_df[~events_df.index.isin(rejected_events)]
                    print(f"  ◐ Partial: {trial_string} {data_type} "
                          f"({len(final_events)}/{len(events_df)} events kept)")
                    
                elif decision == 'reject':
                    # Create empty dataframe with same structure
                    final_events = events_df.iloc[0:0].copy()
                    print(f"  ✗ Rejected: {trial_string} {data_type} (0 events)")
                
                # Save final QC'd events
                final_file = trial_dir / f"events_{data_type}_{trial_string}_QC_final.csv"
                final_events.to_csv(final_file, index=False)
        
        print("Advanced QC application complete!")
    
    def run_advanced_qc_session(self):
        """Main advanced QC session"""
        print("Starting Advanced Interactive Event QC Session")
        print("="*60)
        print("Controls:")
        print("  - Click cells in video to select/deselect")
        print("  - Click events in timeseries to select/deselect")
        print("  - Click timeline to select time ranges")
        print("  - Use buttons for playback, zoom, and QC decisions")
        print("  - Keyboard: Space=play/pause, Left/Right arrows=frame step")
        print("="*60)
        
        # Find all trials
        trials = self.find_trial_files()
        
        if not trials:
            print("No complete trial datasets found!")
            return
        
        print(f"Found {len(trials)} complete trials to review")
        
        # Process each trial for both voltage and calcium
        for trial_string, trial_files in trials.items():
            
            if trial_string not in self.qc_decisions:
                self.qc_decisions[trial_string] = {}
            
            # Process voltage first, then calcium
            for data_type in ['voltage', 'calcium']:
                
                if data_type in self.qc_decisions[trial_string]:
                    print(f"Skipping {trial_string} {data_type} (already reviewed)")
                    continue
                
                print(f"\n{'='*60}")
                print(f"ADVANCED QC: {trial_string} - {data_type.upper()}")
                print(f"{'='*60}")
                
                # Load trial data
                trial_data = self.load_trial_data(trial_string, trial_files, data_type)
                
                if trial_data is None:
                    print(f"Failed to load data for {trial_string} {data_type}")
                    continue
                
                # Create interactive interface
                fig = self.create_interactive_interface(trial_data)
                
                # Show interface (blocks until user closes window)
                plt.show(block=True)
        
        # Save and apply results
        self.save_qc_results()
        
        apply_now = input("\nApply QC decisions now? (y/n): ").strip().lower()
        if apply_now in ['y', 'yes']:
            self.apply_qc_decisions()


def main():
    """Main function - only requires base_results_dir and data_dir"""
    print("Advanced Interactive Event QC Tool")
    home = Path.home()
    if "ys5320" in str(home):
        top_dir = Path(home, "firefly_link")
    else:
        top_dir = Path(r'R:\home\firefly_link')
    
    # Configuration
    base_results_dir = Path(top_dir, r'Calcium_Voltage_Imaging\code_yilin\results_fixed_std')
    data_dir = Path(top_dir, r'ca_voltage_imaging_working\results')
    imaging_base_dir = Path(top_dir, r'Calcium_Voltage_Imaging\MDA_MB_468') 
    
    # Validate paths
    if not base_results_dir.exists():
        print(f"Results directory not found: {base_results_dir}")
        return
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Create QC tool (no imaging_base_dir needed)
    qc_tool = AdvancedInteractiveEventQC(base_results_dir, data_dir, imaging_base_dir)
    
    # Run QC session
    qc_tool.run_advanced_qc_session()


if __name__ == "__main__":
    main()

