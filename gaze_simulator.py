import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import random
from datetime import datetime, timedelta

class EEGViewerWithSimulatedGaze:
    def __init__(self, edf_path, events_data, time_window=5.0, overlap_ratio=0.2,
                 amplitude_scale=245, offset_step=200, heatmap_sigma=10, heatmap_alpha=0.6):
        # ==== Load EDF ====
        self.f = pyedflib.EdfReader(edf_path)
        self.n_signals = self.f.signals_in_file
        self.labels = self.f.getSignalLabels()
        self.sample_rate = int(self.f.getSampleFrequency(0))
        self.signals = [self.f.readSignal(i) for i in range(self.n_signals)]
        self.duration = len(self.signals[0]) / self.sample_rate
        self.f.close()

        print(f"=== EEG INFO ===")
        print(f"EEG Duration: {self.duration:.2f} seconds")
        print(f"EEG Sample Rate: {self.sample_rate} Hz")
        print(f"Number of channels: {self.n_signals}")
        print(f"Channel labels: {self.labels}")

        # ==== Process Events Data ====
        self.events_data = events_data
        self.events = self.process_events_data()
        
        print(f"\n=== EVENTS INFO ===")
        print(f"Loaded {len(self.events)} events")
        
        if self.events:
            # Print detailed info about first 5 events
            print("\nFirst 5 events (relative to File Start):")
            for i, event in enumerate(self.events[:5]):
                print(f"  Event {i}:")
                print(f"    Start: {event['start']:.2f}s")
                print(f"    End: {event['end']:.2f}s") 
                print(f"    Duration: {event['duration']:.2f}s")
                print(f"    Channels: {event['channels']}")
                print(f"    Comment: {event['comment']}")

        # ==== Parameters ====
        self.time_window = time_window
        self.overlap_ratio = overlap_ratio
        self.amplitude_scale = amplitude_scale
        self.offset_step = offset_step
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_alpha = heatmap_alpha

        # ==== Matplotlib figure ====
        plt.ion()
        self.fig, (self.ax, self.cax) = plt.subplots(1, 2, figsize=(16, 7), 
                                                     gridspec_kw={'width_ratios': [15, 1]})
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cbar = None
        self.current_time = 0.0
        self.plot_current_window()
        plt.show(block=True)

    def process_events_data(self):
        """Process events data and map to EDF timeline using File Start as reference"""
        events = []
        
        # Get the file start time from the first row
        file_start_time_str = None
        for _, event in self.events_data.iterrows():
            if pd.notna(event['File Start']):
                file_start_time_str = event['File Start']
                break
        
        if file_start_time_str is None:
            print("Warning: Could not find File Start time in events data")
            # If no file start, assume events start at 0
            file_start_seconds = 0
        else:
            # Convert file start time to seconds
            file_start_seconds = self.time_string_to_seconds(file_start_time_str)
        
        print(f"File Start: {file_start_time_str} = {file_start_seconds:.2f} seconds")
        
        for _, event in self.events_data.iterrows():
            # Skip rows with missing start times
            if pd.isna(event['Start time']):
                continue
                
            # Parse start and end times as seconds
            start_sec = self.time_string_to_seconds(event['Start time'])
            end_sec = self.time_string_to_seconds(event['End time'])
            
            # Convert to relative time (File Start = 0 seconds)
            start_relative = start_sec - file_start_seconds
            end_relative = end_sec - file_start_seconds
            
            # Parse channel names
            channels = event['Channel names'].split()
            
            events.append({
                'start': start_relative,
                'end': end_relative,
                'channels': channels,
                'duration': end_relative - start_relative,
                'comment': event.get('Comment', '')
            })
        
        # Sort events by start time
        events.sort(key=lambda x: x['start'])
        
        # Filter events that fall within EDF duration
        valid_events = [e for e in events if 0 <= e['start'] <= self.duration]
        print(f"Events within EDF duration: {len(valid_events)}/{len(events)}")
        
        if valid_events:
            print(f"First event at: {valid_events[0]['start']:.2f}s")
            print(f"Last event at: {valid_events[-1]['start']:.2f}s")
        
        return valid_events
    
    def time_string_to_seconds(self, time_str):
        """Convert time string to total seconds"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 4:  # HH:MM:SS:ms
                    hours, minutes, seconds, milliseconds = parts
                    total_seconds = (int(hours) * 3600 + 
                                   int(minutes) * 60 + 
                                   int(seconds) + 
                                   int(milliseconds) / 1000)
                elif len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = parts
                    total_seconds = (int(hours) * 3600 + 
                                   int(minutes) * 60 + 
                                   float(seconds))
                else:
                    total_seconds = float(time_str)
            else:
                total_seconds = float(time_str)
                
            return total_seconds
        except Exception as e:
            print(f"Error parsing time string '{time_str}': {e}")
            return 0.0

    def get_events_in_current_window(self):
        """Get events that occur within or overlap with the current time window"""
        window_start = self.current_time
        window_end = self.current_time + self.time_window
        
        events_in_window = []
        for event in self.events:
            # Check if event overlaps with current window
            if (event['start'] <= window_end and event['end'] >= window_start):
                events_in_window.append(event)
        
        return events_in_window

    def generate_gaze_points_using_kde(self, events_in_window):
        """Generate gaze points using proper eye-tracking physics"""
        if not events_in_window:
            return []  # No events = no gaze points
        
        # Eye tracker physics: 33Hz = 33 points per second
        GAZE_SAMPLING_RATE = 33  # Hz
        
        # Calculate total gaze points based on REAL eye tracking
        total_gaze_points = 0
        
        for event in events_in_window:
            # Calculate how much of this event is visible in current window
            window_start = self.current_time
            window_end = self.current_time + self.time_window
            event_overlap_start = max(event['start'], window_start)
            event_overlap_end = min(event['end'], window_end)
            event_overlap_duration = max(0, event_overlap_end - event_overlap_start)
            
            if event_overlap_duration <= 0:
                continue
                
            # Base points: time-based (33 points per second)
            time_based_points = int(event_overlap_duration * GAZE_SAMPLING_RATE)
            
            # Channel multiplier: more channels = more points (spread attention)
            num_channels = len(event['channels'])
            channel_multiplier = min(3.0, 1.0 + (num_channels - 1) * 0.5)  # 1-3x multiplier
            
            event_points = int(time_based_points * channel_multiplier)
            total_gaze_points += event_points
        
        # Ensure reasonable bounds
        total_gaze_points = max(10, min(total_gaze_points, 2000))
        
        print(f"Generating {total_gaze_points} gaze points at {GAZE_SAMPLING_RATE}Hz")
        
        # Create anchor points for KDE based on events
        anchor_times = []
        anchor_channels = []
        
        for event in events_in_window:
            # Calculate how much of this event is visible in current window
            window_start = self.current_time
            window_end = self.current_time + self.time_window
            event_overlap_start = max(event['start'], window_start)
            event_overlap_end = min(event['end'], window_end)
            event_overlap_duration = max(0, event_overlap_end - event_overlap_start)
            
            if event_overlap_duration <= 0:
                continue
                
            # Number of anchors proportional to event importance
            event_duration = event['end'] - event['start']
            num_anchors = max(3, int(event_overlap_duration * 20))  # 20 anchors per second
            
            for _ in range(num_anchors):
                # Time anchors: focus on the overlapping portion
                overlap_center = (event_overlap_start + event_overlap_end) / 2
                time_std = event_overlap_duration / 4  # Spread within overlap region
                time_anchor = np.random.normal(overlap_center, time_std)
                time_anchor = max(event_overlap_start, min(event_overlap_end, time_anchor))
                
                # Channel anchors: prefer event channels
                if event['channels']:
                    channel_anchor = random.choice(event['channels'])
                    if channel_anchor in self.labels:
                        channel_idx = self.labels.index(channel_anchor)
                        # Convert channel to vertical position
                        channel_pos = channel_idx * self.offset_step
                        # Add small vertical variation
                        channel_pos += np.random.normal(0, self.offset_step / 10)
                        
                        anchor_times.append(time_anchor)
                        anchor_channels.append(channel_pos)
        
        if len(anchor_times) < 2:
            # Not enough anchors for KDE, use simple fallback
            return self.generate_gaze_points_simple_fallback(events_in_window, total_gaze_points)
        
        # Prepare data for KDE (2D: time and channel position)
        kde_data = np.vstack([anchor_times, anchor_channels])
        
        try:
            # Create KDE model
            kde = gaussian_kde(kde_data)
            
            # Generate samples from KDE
            samples = kde.resample(total_gaze_points)
            
            gaze_points = []
            for i in range(total_gaze_points):
                time_sample = samples[0, i]
                channel_pos_sample = samples[1, i]
                
                # Ensure samples are within current window
                if (self.current_time <= time_sample <= self.current_time + self.time_window and
                    0 <= channel_pos_sample <= (len(self.labels) - 1) * self.offset_step):
                    
                    # Find closest channel
                    channel_idx = int(round(channel_pos_sample / self.offset_step))
                    channel_idx = max(0, min(channel_idx, len(self.labels) - 1))
                    channel_name = self.labels[channel_idx]
                    
                    # Add small jitter for natural variation
                    time_jitter = np.random.normal(0, 0.05)
                    vertical_jitter = np.random.normal(0, self.offset_step / 15)
                    
                    gaze_points.append({
                        'x': time_sample + time_jitter,
                        'y': channel_pos_sample + vertical_jitter,
                        'time': time_sample,
                        'channel': channel_name,
                        'duration': np.random.uniform(0.2, 0.5)
                    })
            
            return gaze_points
            
        except Exception as e:
            print(f"KDE sampling failed: {e}, using fallback")
            return self.generate_gaze_points_simple_fallback(events_in_window, total_gaze_points)

    def generate_gaze_points_simple_fallback(self, events_in_window, num_points):
        """Fallback method if KDE fails - distribute points by event duration and channels"""
        gaze_points = []
        
        if not events_in_window:
            return []
        
        # Calculate total weighted duration for proportional distribution
        total_weighted_duration = 0
        event_weights = []
        
        for event in events_in_window:
            event_duration = event['end'] - event['start']
            num_channels = len(event['channels'])
            channel_multiplier = min(3.0, 1.0 + (num_channels - 1) * 0.5)
            weighted_duration = event_duration * channel_multiplier
            total_weighted_duration += weighted_duration
            event_weights.append(weighted_duration)
        
        for i, event in enumerate(events_in_window):
            # Points proportional to this event's weighted duration
            weight_fraction = event_weights[i] / total_weighted_duration
            points_for_event = max(5, int(num_points * weight_fraction))
            
            for j in range(points_for_event):
                # Gaussian time distribution around event center
                event_center = (event['start'] + event['end']) / 2
                event_duration = event['end'] - event['start']
                time_std = max(0.5, event_duration / 3)
                gaze_time = np.random.normal(event_center, time_std)
                gaze_time = max(event['start'], min(event['end'], gaze_time))
                
                # Ensure within current window
                if self.current_time <= gaze_time <= self.current_time + self.time_window:
                    # Choose channel from event channels
                    if event['channels']:
                        chosen_channel = random.choice(event['channels'])
                        if chosen_channel in self.labels:
                            channel_idx = self.labels.index(chosen_channel)
                            channel_offset = channel_idx * self.offset_step
                            
                            # Add variation
                            y_variation = np.random.normal(0, self.offset_step / 8)
                            x_jitter = np.random.normal(0, 0.02)
                            
                            gaze_points.append({
                                'x': gaze_time + x_jitter,
                                'y': channel_offset + y_variation,
                                'time': gaze_time,
                                'channel': chosen_channel,
                                'duration': np.random.uniform(0.2, 0.5)
                            })
        
        return gaze_points

    def create_heatmap_for_events(self, gaze_points, window_height):
        """Create a heatmap ONLY if we have gaze points"""
        if not gaze_points:
            return None
            
        # Create a grid for the heatmap
        grid_w, grid_h = 200, 100
        
        heatmap = np.zeros((grid_h, grid_w), dtype=float)
        
        for gaze in gaze_points:
            # Convert time and position to grid coordinates
            time_normalized = (gaze['x'] - self.current_time) / self.time_window
            channel_normalized = gaze['y'] / window_height
            
            x_idx = int(time_normalized * (grid_w - 1))
            y_idx = int(channel_normalized * (grid_h - 1))
            
            if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
                # Use duration as weight for intensity
                heatmap[y_idx, x_idx] += gaze['duration']
        
        # Apply Gaussian smoothing only if we have points
        if np.max(heatmap) > 0:
            heatmap_smooth = gaussian_filter(heatmap, sigma=self.heatmap_sigma)
            # Normalize to 0-1 range
            if np.max(heatmap_smooth) > np.min(heatmap_smooth):
                heatmap_norm = (heatmap_smooth - np.min(heatmap_smooth)) / (np.max(heatmap_smooth) - np.min(heatmap_smooth))
            else:
                heatmap_norm = heatmap_smooth
            return heatmap_norm
        else:
            return None

    def plot_current_window(self):
        self.ax.clear()
        if self.cbar is not None:
            self.cax.clear()
        
        # Ensure current_time is within valid range
        self.current_time = max(0, min(self.current_time, self.duration - self.time_window))
        
        start_idx = int(self.current_time * self.sample_rate)
        end_idx = int((self.current_time + self.time_window) * self.sample_rate)
        end_idx = min(end_idx, len(self.signals[0]))
        
        # Check if we have valid data to plot
        if start_idx >= end_idx:
            print(f"Warning: No data to plot at time {self.current_time}")
            self.ax.text(0.5, 0.5, "No EEG data in this window", 
                        transform=self.ax.transAxes, ha='center', va='center', fontsize=12)
            self.fig.canvas.draw_idle()
            return

        print(f"\n=== Plotting window [{self.current_time:.2f}-{self.current_time+self.time_window:.2f}s] ===")

        # === Plot EEG segment ===
        channel_offsets = {}
        window_height = self.offset_step * len(self.labels)
        
        for i, label in enumerate(self.labels):
            sig = self.signals[i]
            segment = sig[start_idx:end_idx]
            
            # Only plot if we have data
            if len(segment) > 0:
                times = np.linspace(self.current_time, self.current_time + self.time_window, len(segment))
                offset = i * self.offset_step
                channel_offsets[label] = offset
                
                self.ax.plot(times, segment * self.amplitude_scale + offset, color='black', linewidth=0.8, zorder=3)
                
                # Only add text if we have times data
                if len(times) > 0:
                    self.ax.text(times[0] + 0.05, offset, label,
                                va='center', ha='left', fontsize=8, fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.9), zorder=4)
                else:
                    self.ax.text(self.current_time + 0.05, offset, label,
                                va='center', ha='left', fontsize=8, fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.9), zorder=4)

        # === Get events in current window and generate gaze points ===
        events_in_window = self.get_events_in_current_window()
        print(f"Events in current window: {len(events_in_window)}")
        
        # ONLY generate gaze points if there are events in this window
        gaze_points = []
        if events_in_window:
            gaze_points = self.generate_gaze_points_using_kde(events_in_window)
            print(f"Generated {len(gaze_points)} gaze points")
        else:
            print("No events in window - no gaze points generated")
        
        # === Create and plot heatmap ONLY if we have gaze points ===
        heatmap = None
        if gaze_points:
            heatmap = self.create_heatmap_for_events(gaze_points, window_height)
        
        if heatmap is not None:
            # Define extent for the heatmap (covers entire current window)
            extent = [
                self.current_time, 
                self.current_time + self.time_window, 
                0, 
                window_height
            ]
            
            # Plot the heatmap behind the EEG signals
            im = self.ax.imshow(
                heatmap,
                extent=extent,
                origin='lower',
                cmap='hot',
                alpha=self.heatmap_alpha,
                aspect='auto',
                zorder=2
            )
            
            # Create/update colorbar
            if self.cbar is None:
                self.cbar = self.fig.colorbar(im, cax=self.cax)
                self.cbar.set_label('Gaze Intensity', fontsize=9)
            else:
                self.cax.clear()
                self.cbar = self.fig.colorbar(im, cax=self.cax)
                self.cbar.set_label('Gaze Intensity', fontsize=9)

        # === Plot individual gaze points ONLY if we have them ===
        # if gaze_points:
        #     self.ax.scatter([g['x'] for g in gaze_points], [g['y'] for g in gaze_points],
        #                    c='red', s=20, alpha=0.7, zorder=4, label='Event Gaze')
        #     self.ax.legend(loc='upper right', fontsize=8)

        # === Scale bar ===
        bar_x = self.current_time + self.time_window - 0.5
        bar_y = -80
        self.ax.plot([bar_x, bar_x], [bar_y, bar_y + 50], color='black', lw=2, zorder=4)
        self.ax.plot([bar_x, bar_x + 0.2], [bar_y, bar_y], color='black', lw=2, zorder=4)
        self.ax.text(bar_x + 0.05, bar_y + 25, "50 µV", va='center', fontsize=8, zorder=4)
        self.ax.text(bar_x + 0.1, bar_y - 8, "0.2 s", ha='center', fontsize=8, zorder=4)

        # === Style ===
        self.ax.set_xlim(self.current_time, self.current_time + self.time_window)
        self.ax.set_ylim(-100, window_height)
        self.ax.get_yaxis().set_visible(False)
        
        # Update title based on whether we have events
        if events_in_window:
            self.ax.set_title(
                f"EEG + Event Gaze Heatmap | Time: {self.current_time:.1f}-{self.current_time+self.time_window:.1f}s | {len(events_in_window)} events, {len(gaze_points)} gaze points",
                fontsize=12, weight='bold'
            )
        else:
            self.ax.set_title(
                f"EEG Only | Time: {self.current_time:.1f}-{self.current_time+self.time_window:.1f}s | No events in window",
                fontsize=12, weight='bold'
            )

        # Hide the colorbar axis if no heatmap
        if heatmap is None:
            self.cax.set_visible(False)
        else:
            self.cax.set_visible(True)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def scroll_time(self, forward=True):
        step = self.time_window * (1 - self.overlap_ratio)
        if forward:
            new_time = self.current_time + step
            if new_time < self.duration - self.time_window:
                self.current_time = new_time
            else:
                self.current_time = max(0, self.duration - self.time_window)
        else:
            new_time = self.current_time - step
            self.current_time = max(0, new_time)
        self.plot_current_window()

    def on_key_press(self, event):
        if event.key == 'right':
            self.scroll_time(forward=True)
        elif event.key == 'left':
            self.scroll_time(forward=False)
        elif event.key == 'home':
            self.current_time = 0.0
            self.plot_current_window()
        elif event.key == 'end':
            self.current_time = max(0, self.duration - self.time_window)
            self.plot_current_window()
        elif event.key == 'up':
            # Jump to first event time
            if self.events:
                first_event = min(self.events, key=lambda x: x['start'])
                self.current_time = max(0, first_event['start'] - self.time_window/2)
                self.plot_current_window()
        elif event.key == 'down':
            # Jump to last event time
            if self.events:
                last_event = max(self.events, key=lambda x: x['end'])
                self.current_time = max(0, last_event['end'] - self.time_window/2)
                self.plot_current_window()

# Example usage
if __name__ == "__main__":
    # Load your events data from CSV
    events_df = pd.read_csv(r"C:\Users\S.S.T\Downloads\SW & SSW CSV Files\SW & SSW CSV Files\6.csv")
    
    # Fill empty demographic cells
    events_df['Gender'] = events_df['Gender'].fillna(method='ffill')
    events_df['Age'] = events_df['Age'].fillna(method='ffill') 
    events_df['File Start'] = events_df['File Start'].fillna(method='ffill')
    
    EEGViewerWithSimulatedGaze(
        edf_path=r"C:\Users\S.S.T\Downloads\0000006.edf",
        events_data=events_df,
        time_window=5.0,
        overlap_ratio=0.2,
        heatmap_sigma=8,
        heatmap_alpha=0.7
    )