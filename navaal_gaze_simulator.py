# import pyedflib
# import numpy as np
# import matplotlib.pyplot as plt
# import json
# import pandas as pd
# from scipy.ndimage import gaussian_filter
# from scipy.stats import gaussian_kde
# import random
# from datetime import datetime, timedelta

# class EEGViewerWithSimulatedGaze:
#     def __init__(self, edf_path, events_data, time_window=5.0, overlap_ratio=0.2,
#                  amplitude_scale=245, offset_step=200, heatmap_sigma=10, heatmap_alpha=0.6):
#         # ==== Load EDF ====
#         self.f = pyedflib.EdfReader(edf_path)
#         self.n_signals = self.f.signals_in_file
#         self.labels = self.f.getSignalLabels()
#         self.sample_rate = int(self.f.getSampleFrequency(0))
#         self.signals = [self.f.readSignal(i) for i in range(self.n_signals)]
#         self.duration = len(self.signals[0]) / self.sample_rate
#         self.f.close()

#         print(f"=== EEG INFO ===")
#         print(f"EEG Duration: {self.duration:.2f} seconds")
#         print(f"EEG Sample Rate: {self.sample_rate} Hz")
#         print(f"Number of channels: {self.n_signals}")
#         print(f"Channel labels: {self.labels}")

#         # ==== Process Events Data ====
#         self.events_data = events_data
#         self.events = self.process_events_data()
        
#         print(f"\n=== EVENTS INFO ===")
#         print(f"Loaded {len(self.events)} events")
        
#         if self.events:
#             # Print detailed info about first 5 events
#             print("\nFirst 5 events (relative to File Start):")
#             for i, event in enumerate(self.events[:5]):
#                 print(f"  Event {i}:")
#                 print(f"    Start: {event['start']:.2f}s")
#                 print(f"    End: {event['end']:.2f}s") 
#                 print(f"    Duration: {event['duration']:.2f}s")
#                 print(f"    Channels: {event['channels']}")
#                 print(f"    Comment: {event['comment']}")

#         # ==== Parameters ====
#         self.time_window = time_window
#         self.overlap_ratio = overlap_ratio
#         self.amplitude_scale = amplitude_scale
#         self.offset_step = offset_step
#         self.heatmap_sigma = heatmap_sigma
#         self.heatmap_alpha = heatmap_alpha

#         # ==== Matplotlib figure ====
#         plt.ion()
#         self.fig, (self.ax, self.cax) = plt.subplots(1, 2, figsize=(16, 7), 
#                                                      gridspec_kw={'width_ratios': [15, 1]})
#         self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
#         self.cbar = None
#         self.current_time = 0.0
#         self.plot_current_window()
#         plt.show(block=True)

#     def process_events_data(self):
#         """Process events data and map to EDF timeline using File Start as reference"""
#         events = []
        
#         # Get the file start time from the first row
#         file_start_time_str = None
#         for _, event in self.events_data.iterrows():
#             if pd.notna(event['File Start']):
#                 file_start_time_str = event['File Start']
#                 break
        
#         if file_start_time_str is None:
#             print("Warning: Could not find File Start time in events data")
#             # If no file start, assume events start at 0
#             file_start_seconds = 0
#         else:
#             # Convert file start time to seconds
#             file_start_seconds = self.time_string_to_seconds(file_start_time_str)
        
#         print(f"File Start: {file_start_time_str} = {file_start_seconds:.2f} seconds")
        
#         for _, event in self.events_data.iterrows():
#             # Skip rows with missing start times
#             if pd.isna(event['Start time']):
#                 continue
                
#             # Parse start and end times as seconds
#             start_sec = self.time_string_to_seconds(event['Start time'])
#             end_sec = self.time_string_to_seconds(event['End time'])
            
#             # Convert to relative time (File Start = 0 seconds)
#             start_relative = start_sec - file_start_seconds
#             end_relative = end_sec - file_start_seconds
            
#             # Parse channel names
#             channels = event['Channel names'].split()
            
#             events.append({
#                 'start': start_relative,
#                 'end': end_relative,
#                 'channels': channels,
#                 'duration': end_relative - start_relative,
#                 'comment': event.get('Comment', '')
#             })
        
#         # Sort events by start time
#         events.sort(key=lambda x: x['start'])
        
#         # Filter events that fall within EDF duration
#         valid_events = [e for e in events if 0 <= e['start'] <= self.duration]
#         print(f"Events within EDF duration: {len(valid_events)}/{len(events)}")
        
#         if valid_events:
#             print(f"First event at: {valid_events[0]['start']:.2f}s")
#             print(f"Last event at: {valid_events[-1]['start']:.2f}s")
        
#         return valid_events
    
#     def time_string_to_seconds(self, time_str):
#         """Convert time string to total seconds"""
#         try:
#             if ':' in time_str:
#                 parts = time_str.split(':')
#                 if len(parts) == 4:  # HH:MM:SS:ms
#                     hours, minutes, seconds, milliseconds = parts
#                     total_seconds = (int(hours) * 3600 + 
#                                    int(minutes) * 60 + 
#                                    int(seconds) + 
#                                    int(milliseconds) / 1000)
#                 elif len(parts) == 3:  # HH:MM:SS
#                     hours, minutes, seconds = parts
#                     total_seconds = (int(hours) * 3600 + 
#                                    int(minutes) * 60 + 
#                                    float(seconds))
#                 else:
#                     total_seconds = float(time_str)
#             else:
#                 total_seconds = float(time_str)
                
#             return total_seconds
#         except Exception as e:
#             print(f"Error parsing time string '{time_str}': {e}")
#             return 0.0

#     def get_events_in_current_window(self):
#         """Get events that occur within or overlap with the current time window"""
#         window_start = self.current_time
#         window_end = self.current_time + self.time_window
        
#         events_in_window = []
#         for event in self.events:
#             # Check if event overlaps with current window
#             if (event['start'] <= window_end and event['end'] >= window_start):
#                 events_in_window.append(event)
        
#         return events_in_window

#     def generate_gaze_points_using_kde(self, events_in_window):
#         """Generate gaze points using proper eye-tracking physics"""
#         if not events_in_window:
#             return []  # No events = no gaze points
        
#         # Eye tracker physics: 33Hz = 33 points per second
#         GAZE_SAMPLING_RATE = 33  # Hz
        
#         # Calculate total gaze points based on REAL eye tracking
#         total_gaze_points = 0
        
#         for event in events_in_window:
#             # Calculate how much of this event is visible in current window
#             window_start = self.current_time
#             window_end = self.current_time + self.time_window
#             event_overlap_start = max(event['start'], window_start)
#             event_overlap_end = min(event['end'], window_end)
#             event_overlap_duration = max(0, event_overlap_end - event_overlap_start)
            
#             if event_overlap_duration <= 0:
#                 continue
                
#             # Base points: time-based (33 points per second)
#             time_based_points = int(event_overlap_duration * GAZE_SAMPLING_RATE)
            
#             # Channel multiplier: more channels = more points (spread attention)
#             num_channels = len(event['channels'])
#             channel_multiplier = min(3.0, 1.0 + (num_channels - 1) * 0.5)  # 1-3x multiplier
            
#             event_points = int(time_based_points * channel_multiplier)
#             total_gaze_points += event_points
        
#         # Ensure reasonable bounds
#         total_gaze_points = max(10, min(total_gaze_points, 2000))
        
#         print(f"Generating {total_gaze_points} gaze points at {GAZE_SAMPLING_RATE}Hz")
        
#         # Create anchor points for KDE based on events
#         anchor_times = []
#         anchor_channels = []
        
#         for event in events_in_window:
#             # Calculate how much of this event is visible in current window
#             window_start = self.current_time
#             window_end = self.current_time + self.time_window
#             event_overlap_start = max(event['start'], window_start)
#             event_overlap_end = min(event['end'], window_end)
#             event_overlap_duration = max(0, event_overlap_end - event_overlap_start)
            
#             if event_overlap_duration <= 0:
#                 continue
                
#             # Number of anchors proportional to event importance
#             event_duration = event['end'] - event['start']
#             num_anchors = max(3, int(event_overlap_duration * 20))  # 20 anchors per second
            
#             for _ in range(num_anchors):
#                 # Time anchors: focus on the overlapping portion
#                 overlap_center = (event_overlap_start + event_overlap_end) / 2
#                 time_std = event_overlap_duration / 4  # Spread within overlap region
#                 time_anchor = np.random.normal(overlap_center, time_std)
#                 time_anchor = max(event_overlap_start, min(event_overlap_end, time_anchor))
                
#                 # Channel anchors: prefer event channels
#                 if event['channels']:
#                     channel_anchor = random.choice(event['channels'])
#                     if channel_anchor in self.labels:
#                         channel_idx = self.labels.index(channel_anchor)
#                         # Convert channel to vertical position
#                         channel_pos = channel_idx * self.offset_step
#                         # Add small vertical variation
#                         channel_pos += np.random.normal(0, self.offset_step / 10)
                        
#                         anchor_times.append(time_anchor)
#                         anchor_channels.append(channel_pos)
        
#         if len(anchor_times) < 2:
#             # Not enough anchors for KDE, use simple fallback
#             return self.generate_gaze_points_simple_fallback(events_in_window, total_gaze_points)
        
#         # Prepare data for KDE (2D: time and channel position)
#         kde_data = np.vstack([anchor_times, anchor_channels])
        
#         try:
#             # Create KDE model
#             kde = gaussian_kde(kde_data)
            
#             # Generate samples from KDE
#             samples = kde.resample(total_gaze_points)
            
#             gaze_points = []
#             for i in range(total_gaze_points):
#                 time_sample = samples[0, i]
#                 channel_pos_sample = samples[1, i]
                
#                 # Ensure samples are within current window
#                 if (self.current_time <= time_sample <= self.current_time + self.time_window and
#                     0 <= channel_pos_sample <= (len(self.labels) - 1) * self.offset_step):
                    
#                     # Find closest channel
#                     channel_idx = int(round(channel_pos_sample / self.offset_step))
#                     channel_idx = max(0, min(channel_idx, len(self.labels) - 1))
#                     channel_name = self.labels[channel_idx]
                    
#                     # Add small jitter for natural variation
#                     time_jitter = np.random.normal(0, 0.05)
#                     vertical_jitter = np.random.normal(0, self.offset_step / 15)
                    
#                     gaze_points.append({
#                         'x': time_sample + time_jitter,
#                         'y': channel_pos_sample + vertical_jitter,
#                         'time': time_sample,
#                         'channel': channel_name,
#                         'duration': np.random.uniform(0.2, 0.5)
#                     })
            
#             return gaze_points
            
#         except Exception as e:
#             print(f"KDE sampling failed: {e}, using fallback")
#             return self.generate_gaze_points_simple_fallback(events_in_window, total_gaze_points)

#     def generate_gaze_points_simple_fallback(self, events_in_window, num_points):
#         """Fallback method if KDE fails - distribute points by event duration and channels"""
#         gaze_points = []
        
#         if not events_in_window:
#             return []
        
#         # Calculate total weighted duration for proportional distribution
#         total_weighted_duration = 0
#         event_weights = []
        
#         for event in events_in_window:
#             event_duration = event['end'] - event['start']
#             num_channels = len(event['channels'])
#             channel_multiplier = min(3.0, 1.0 + (num_channels - 1) * 0.5)
#             weighted_duration = event_duration * channel_multiplier
#             total_weighted_duration += weighted_duration
#             event_weights.append(weighted_duration)
        
#         for i, event in enumerate(events_in_window):
#             # Points proportional to this event's weighted duration
#             weight_fraction = event_weights[i] / total_weighted_duration
#             points_for_event = max(5, int(num_points * weight_fraction))
            
#             for j in range(points_for_event):
#                 # Gaussian time distribution around event center
#                 event_center = (event['start'] + event['end']) / 2
#                 event_duration = event['end'] - event['start']
#                 time_std = max(0.5, event_duration / 3)
#                 gaze_time = np.random.normal(event_center, time_std)
#                 gaze_time = max(event['start'], min(event['end'], gaze_time))
                
#                 # Ensure within current window
#                 if self.current_time <= gaze_time <= self.current_time + self.time_window:
#                     # Choose channel from event channels
#                     if event['channels']:
#                         chosen_channel = random.choice(event['channels'])
#                         if chosen_channel in self.labels:
#                             channel_idx = self.labels.index(chosen_channel)
#                             channel_offset = channel_idx * self.offset_step
                            
#                             # Add variation
#                             y_variation = np.random.normal(0, self.offset_step / 8)
#                             x_jitter = np.random.normal(0, 0.02)
                            
#                             gaze_points.append({
#                                 'x': gaze_time + x_jitter,
#                                 'y': channel_offset + y_variation,
#                                 'time': gaze_time,
#                                 'channel': chosen_channel,
#                                 'duration': np.random.uniform(0.2, 0.5)
#                             })
        
#         return gaze_points

#     def create_heatmap_for_events(self, gaze_points, window_height):
#         """Create a heatmap ONLY if we have gaze points"""
#         if not gaze_points:
#             return None
            
#         # Create a grid for the heatmap
#         grid_w, grid_h = 200, 100
        
#         heatmap = np.zeros((grid_h, grid_w), dtype=float)
        
#         for gaze in gaze_points:
#             # Convert time and position to grid coordinates
#             time_normalized = (gaze['x'] - self.current_time) / self.time_window
#             channel_normalized = gaze['y'] / window_height
            
#             x_idx = int(time_normalized * (grid_w - 1))
#             y_idx = int(channel_normalized * (grid_h - 1))
            
#             if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
#                 # Use duration as weight for intensity
#                 heatmap[y_idx, x_idx] += gaze['duration']
        
#         # Apply Gaussian smoothing only if we have points
#         if np.max(heatmap) > 0:
#             heatmap_smooth = gaussian_filter(heatmap, sigma=self.heatmap_sigma)
#             # Normalize to 0-1 range
#             if np.max(heatmap_smooth) > np.min(heatmap_smooth):
#                 heatmap_norm = (heatmap_smooth - np.min(heatmap_smooth)) / (np.max(heatmap_smooth) - np.min(heatmap_smooth))
#             else:
#                 heatmap_norm = heatmap_smooth
#             return heatmap_norm
#         else:
#             return None

#     def plot_current_window(self):
#         self.ax.clear()
#         if self.cbar is not None:
#             self.cax.clear()
        
#         # Ensure current_time is within valid range
#         self.current_time = max(0, min(self.current_time, self.duration - self.time_window))
        
#         start_idx = int(self.current_time * self.sample_rate)
#         end_idx = int((self.current_time + self.time_window) * self.sample_rate)
#         end_idx = min(end_idx, len(self.signals[0]))
        
#         # Check if we have valid data to plot
#         if start_idx >= end_idx:
#             print(f"Warning: No data to plot at time {self.current_time}")
#             self.ax.text(0.5, 0.5, "No EEG data in this window", 
#                         transform=self.ax.transAxes, ha='center', va='center', fontsize=12)
#             self.fig.canvas.draw_idle()
#             return

#         print(f"\n=== Plotting window [{self.current_time:.2f}-{self.current_time+self.time_window:.2f}s] ===")

#         # === Plot EEG segment ===
#         channel_offsets = {}
#         window_height = self.offset_step * len(self.labels)
        
#         for i, label in enumerate(self.labels):
#             sig = self.signals[i]
#             segment = sig[start_idx:end_idx]
            
#             # Only plot if we have data
#             if len(segment) > 0:
#                 times = np.linspace(self.current_time, self.current_time + self.time_window, len(segment))
#                 offset = i * self.offset_step
#                 channel_offsets[label] = offset
                
#                 self.ax.plot(times, segment * self.amplitude_scale + offset, color='black', linewidth=0.8, zorder=3)
                
#                 # Only add text if we have times data
#                 if len(times) > 0:
#                     self.ax.text(times[0] + 0.05, offset, label,
#                                 va='center', ha='left', fontsize=8, fontweight='bold',
#                                 bbox=dict(facecolor='white', alpha=0.9), zorder=4)
#                 else:
#                     self.ax.text(self.current_time + 0.05, offset, label,
#                                 va='center', ha='left', fontsize=8, fontweight='bold',
#                                 bbox=dict(facecolor='white', alpha=0.9), zorder=4)

#         # === Get events in current window and generate gaze points ===
#         events_in_window = self.get_events_in_current_window()
#         print(f"Events in current window: {len(events_in_window)}")
        
#         # ONLY generate gaze points if there are events in this window
#         gaze_points = []
#         if events_in_window:
#             gaze_points = self.generate_gaze_points_using_kde(events_in_window)
#             print(f"Generated {len(gaze_points)} gaze points")
#         else:
#             print("No events in window - no gaze points generated")
        
#         # === Create and plot heatmap ONLY if we have gaze points ===
#         heatmap = None
#         if gaze_points:
#             heatmap = self.create_heatmap_for_events(gaze_points, window_height)
        
#         if heatmap is not None:
#             # Define extent for the heatmap (covers entire current window)
#             extent = [
#                 self.current_time, 
#                 self.current_time + self.time_window, 
#                 0, 
#                 window_height
#             ]
            
#             # Plot the heatmap behind the EEG signals
#             im = self.ax.imshow(
#                 heatmap,
#                 extent=extent,
#                 origin='lower',
#                 cmap='hot',
#                 alpha=self.heatmap_alpha,
#                 aspect='auto',
#                 zorder=2
#             )
            
#             # Create/update colorbar
#             if self.cbar is None:
#                 self.cbar = self.fig.colorbar(im, cax=self.cax)
#                 self.cbar.set_label('Gaze Intensity', fontsize=9)
#             else:
#                 self.cax.clear()
#                 self.cbar = self.fig.colorbar(im, cax=self.cax)
#                 self.cbar.set_label('Gaze Intensity', fontsize=9)

#         # === Plot individual gaze points ONLY if we have them ===
#         # if gaze_points:
#         #     self.ax.scatter([g['x'] for g in gaze_points], [g['y'] for g in gaze_points],
#         #                    c='red', s=20, alpha=0.7, zorder=4, label='Event Gaze')
#         #     self.ax.legend(loc='upper right', fontsize=8)

#         # === Scale bar ===
#         bar_x = self.current_time + self.time_window - 0.5
#         bar_y = -80
#         self.ax.plot([bar_x, bar_x], [bar_y, bar_y + 50], color='black', lw=2, zorder=4)
#         self.ax.plot([bar_x, bar_x + 0.2], [bar_y, bar_y], color='black', lw=2, zorder=4)
#         self.ax.text(bar_x + 0.05, bar_y + 25, "50 µV", va='center', fontsize=8, zorder=4)
#         self.ax.text(bar_x + 0.1, bar_y - 8, "0.2 s", ha='center', fontsize=8, zorder=4)

#         # === Style ===
#         self.ax.set_xlim(self.current_time, self.current_time + self.time_window)
#         self.ax.set_ylim(-100, window_height)
#         self.ax.get_yaxis().set_visible(False)
        
#         # Update title based on whether we have events
#         if events_in_window:
#             self.ax.set_title(
#                 f"EEG + Event Gaze Heatmap | Time: {self.current_time:.1f}-{self.current_time+self.time_window:.1f}s | {len(events_in_window)} events, {len(gaze_points)} gaze points",
#                 fontsize=12, weight='bold'
#             )
#         else:
#             self.ax.set_title(
#                 f"EEG Only | Time: {self.current_time:.1f}-{self.current_time+self.time_window:.1f}s | No events in window",
#                 fontsize=12, weight='bold'
#             )

#         # Hide the colorbar axis if no heatmap
#         if heatmap is None:
#             self.cax.set_visible(False)
#         else:
#             self.cax.set_visible(True)

#         self.fig.canvas.draw_idle()
#         self.fig.canvas.flush_events()

#     def scroll_time(self, forward=True):
#         step = self.time_window * (1 - self.overlap_ratio)
#         if forward:
#             new_time = self.current_time + step
#             if new_time < self.duration - self.time_window:
#                 self.current_time = new_time
#             else:
#                 self.current_time = max(0, self.duration - self.time_window)
#         else:
#             new_time = self.current_time - step
#             self.current_time = max(0, new_time)
#         self.plot_current_window()

#     def on_key_press(self, event):
#         if event.key == 'right':
#             self.scroll_time(forward=True)
#         elif event.key == 'left':
#             self.scroll_time(forward=False)
#         elif event.key == 'home':
#             self.current_time = 0.0
#             self.plot_current_window()
#         elif event.key == 'end':
#             self.current_time = max(0, self.duration - self.time_window)
#             self.plot_current_window()
#         elif event.key == 'up':
#             # Jump to first event time
#             if self.events:
#                 first_event = min(self.events, key=lambda x: x['start'])
#                 self.current_time = max(0, first_event['start'] - self.time_window/2)
#                 self.plot_current_window()
#         elif event.key == 'down':
#             # Jump to last event time
#             if self.events:
#                 last_event = max(self.events, key=lambda x: x['end'])
#                 self.current_time = max(0, last_event['end'] - self.time_window/2)
#                 self.plot_current_window()

# # Example usage
# if __name__ == "__main__":
#     # Load your events data from CSV
#     events_df = pd.read_csv(r"C:\Users\S.S.T\Downloads\SW & SSW CSV Files\SW & SSW CSV Files\6.csv")
    
#     # Fill empty demographic cells
#     events_df['Gender'] = events_df['Gender'].fillna(method='ffill')
#     events_df['Age'] = events_df['Age'].fillna(method='ffill') 
#     events_df['File Start'] = events_df['File Start'].fillna(method='ffill')
    
#     EEGViewerWithSimulatedGaze(
#         edf_path=r"C:\Users\S.S.T\Downloads\0000006.edf",
#         events_data=events_df,
#         time_window=5.0,
#         overlap_ratio=0.2,
#         heatmap_sigma=8,
#         heatmap_alpha=0.7
#     )




    # SECOND VARIANT: Simulating abnormal and normal dummy gaze data
# import pyedflib
# import numpy as np
# import matplotlib.pyplot as plt
# import json
# import pandas as pd
# from scipy.ndimage import gaussian_filter
# from scipy.stats import gaussian_kde
# import random
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# class EEGViewerWithDualGaze:
#     def __init__(self, edf_path, events_data, time_window=5.0, overlap_ratio=0.2,
#                  amplitude_scale=245, offset_step=200, heatmap_sigma=10, heatmap_alpha=0.6):
#         # ==== Load EDF ====
#         self.f = pyedflib.EdfReader(edf_path)
#         self.n_signals = self.f.signals_in_file
#         self.labels = self.f.getSignalLabels()
#         self.sample_rate = int(self.f.getSampleFrequency(0))
#         self.signals = [self.f.readSignal(i) for i in range(self.n_signals)]
#         self.duration = len(self.signals[0]) / self.sample_rate
#         self.f.close()

#         print(f"=== EEG INFO ===")
#         print(f"EEG Duration: {self.duration:.2f} seconds")
#         print(f"EEG Sample Rate: {self.sample_rate} Hz")
#         print(f"Number of channels: {self.n_signals}")
#         print(f"Channel labels: {self.labels}")

#         # ==== Process Events Data ====
#         self.events_data = events_data
#         self.events = self.process_events_data()
        
#         print(f"\n=== EVENTS INFO ===")
#         print(f"Loaded {len(self.events)} abnormal events")
        
#         if self.events:
#             print("\nFirst 5 abnormal events:")
#             for i, event in enumerate(self.events[:5]):
#                 print(f"  Event {i}: {event['start']:.2f}-{event['end']:.2f}s ({event['duration']:.2f}s) on {event['channels']}")

#         # ==== Parameters ====
#         self.time_window = time_window
#         self.overlap_ratio = overlap_ratio
#         self.amplitude_scale = amplitude_scale
#         self.offset_step = offset_step
#         self.heatmap_sigma = heatmap_sigma
#         self.heatmap_alpha = heatmap_alpha
        
#         # Gaze simulation parameters
#         self.GAZE_SAMPLING_RATE = 33  # Hz - typical eye tracker
#         self.NORMAL_GAZE_DENSITY = 0.3  # Lower density for normal periods
#         self.EVENT_GAZE_DENSITY = 1.0   # Higher density for abnormal periods

#         # ==== Matplotlib figure ====
#         plt.ion()
#         self.fig, (self.ax, self.cax) = plt.subplots(1, 2, figsize=(16, 7), 
#                                                      gridspec_kw={'width_ratios': [15, 1]})
#         self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
#         self.cbar = None
#         self.current_time = 0.0
#         self.plot_current_window()
#         plt.show(block=True)

#     def process_events_data(self):
#         """Process events data and map to EDF timeline using File Start as reference"""
#         events = []
        
#         # Get the file start time from the first row
#         file_start_time_str = None
#         for _, event in self.events_data.iterrows():
#             if pd.notna(event['File Start']):
#                 file_start_time_str = event['File Start']
#                 break
        
#         if file_start_time_str is None:
#             print("Warning: Could not find File Start time in events data")
#             file_start_seconds = 0
#         else:
#             file_start_seconds = self.time_string_to_seconds(file_start_time_str)
        
#         print(f"File Start: {file_start_time_str} = {file_start_seconds:.2f} seconds")
        
#         for _, event in self.events_data.iterrows():
#             # Skip rows with missing start times
#             if pd.isna(event['Start time']):
#                 continue
                
#             start_sec = self.time_string_to_seconds(event['Start time'])
#             end_sec = self.time_string_to_seconds(event['End time'])
            
#             # Convert to relative time (File Start = 0 seconds)
#             start_relative = start_sec - file_start_seconds
#             end_relative = end_sec - file_start_seconds
            
#             # Parse channel names
#             channels = event['Channel names'].split()
            
#             events.append({
#                 'start': start_relative,
#                 'end': end_relative,
#                 'channels': channels,
#                 'duration': end_relative - start_relative,
#                 'type': 'abnormal',
#                 'comment': event.get('Comment', '')
#             })
        
#         events.sort(key=lambda x: x['start'])
#         valid_events = [e for e in events if 0 <= e['start'] <= self.duration]
#         print(f"Valid events within EDF duration: {len(valid_events)}/{len(events)}")
        
#         return valid_events
    
#     def time_string_to_seconds(self, time_str):
#         """Convert time string to total seconds"""
#         try:
#             if ':' in time_str:
#                 parts = time_str.split(':')
#                 if len(parts) == 4:  # HH:MM:SS:ms
#                     hours, minutes, seconds, milliseconds = parts
#                     total_seconds = (int(hours) * 3600 + 
#                                    int(minutes) * 60 + 
#                                    int(seconds) + 
#                                    int(milliseconds) / 1000)
#                 elif len(parts) == 3:  # HH:MM:SS
#                     hours, minutes, seconds = parts
#                     total_seconds = (int(hours) * 3600 + 
#                                    int(minutes) * 60 + 
#                                    float(seconds))
#                 else:
#                     total_seconds = float(time_str)
#             else:
#                 total_seconds = float(time_str)
                
#             return total_seconds
#         except Exception as e:
#             print(f"Error parsing time string '{time_str}': {e}")
#             return 0.0

#     def get_normal_periods_in_window(self, window_start, window_end, events_in_window):
#         """Identify normal (non-event) periods within the current window"""
#         normal_periods = []
        
#         if not events_in_window:
#             # No events in window = entire window is normal
#             normal_periods.append({
#                 'start': window_start,
#                 'end': window_end,
#                 'duration': window_end - window_start,
#                 'type': 'normal'
#             })
#         else:
#             # Check gaps between events
#             current_time = window_start
            
#             for event in sorted(events_in_window, key=lambda x: x['start']):
#                 if event['start'] > current_time:
#                     # Found a normal period before this event
#                     normal_periods.append({
#                         'start': current_time,
#                         'end': event['start'],
#                         'duration': event['start'] - current_time,
#                         'type': 'normal'
#                     })
#                 current_time = max(current_time, event['end'])
            
#             # Check for normal period after last event
#             if current_time < window_end:
#                 normal_periods.append({
#                     'start': current_time,
#                     'end': window_end,
#                     'duration': window_end - current_time,
#                     'type': 'normal'
#                 })
        
#         # Filter out very short periods (< 0.1s)
#         normal_periods = [p for p in normal_periods if p['duration'] >= 0.1]
        
#         return normal_periods

#     def generate_normal_gaze_points(self, normal_periods, window_height):
#         """Generate gaze points for normal (non-event) periods"""
#         all_gaze_points = []
        
#         for period in normal_periods:
#             # Calculate number of gaze points based on duration and density
#             total_points = int(period['duration'] * self.GAZE_SAMPLING_RATE * self.NORMAL_GAZE_DENSITY)
            
#             # Ensure minimum points for visualization
#             total_points = max(5, min(total_points, 500))
            
#             for _ in range(total_points):
#                 # Random time within normal period
#                 gaze_time = np.random.uniform(period['start'], period['end'])
                
#                 # Different gaze patterns for normal periods:
#                 # 1. Random scanning across all channels
#                 if np.random.random() < 0.6:  # 60%: random scanning
#                     channel_idx = np.random.randint(0, len(self.labels))
#                 # 2. Focus on central channels (common resting state)
#                 else:  # 40%: focus on central/sensorimotor channels
#                     central_chars = ['C', 'F', 'P']  # Central, Frontal, Parietal
#                     central_channels = [i for i, label in enumerate(self.labels) 
#                                       if any(label.startswith(char) for char in central_chars)]
#                     if central_channels:
#                         channel_idx = np.random.choice(central_channels)
#                     else:
#                         channel_idx = np.random.randint(0, len(self.labels))
                
#                 channel_offset = channel_idx * self.offset_step
#                 channel_name = self.labels[channel_idx]
                
#                 # Add natural variation
#                 y_variation = np.random.normal(0, self.offset_step / 12)
#                 x_jitter = np.random.normal(0, 0.03)  # More jitter in normal scanning
                
#                 # Normal gaze tends to be shorter fixations during scanning
#                 duration = np.random.uniform(0.1, 0.3)  # Shorter for normal
                
#                 gaze_point = {
#                     'x': gaze_time + x_jitter,
#                     'y': channel_offset + y_variation,
#                     'time': gaze_time,
#                     'channel': channel_name,
#                     'duration': duration,
#                     'type': 'normal'
#                 }
#                 all_gaze_points.append(gaze_point)
        
#         return all_gaze_points

#     def generate_abnormal_gaze_points(self, events_in_window, window_height):
#         """Generate gaze points for abnormal (event) periods"""
#         if not events_in_window:
#             return []
        
#         all_gaze_points = []
        
#         for event in events_in_window:
#             # Calculate number of gaze points based on duration and density
#             event_points = int(event['duration'] * self.GAZE_SAMPLING_RATE * self.EVENT_GAZE_DENSITY)
#             event_points = max(10, min(event_points, 1000))
            
#             # Event-specific parameters
#             event_center = (event['start'] + event['end']) / 2
#             event_channels = event['channels']
            
#             # Convert channel names to indices
#             channel_indices = []
#             for ch_name in event_channels:
#                 if ch_name in self.labels:
#                     channel_indices.append(self.labels.index(ch_name))
            
#             if not channel_indices:
#                 continue
            
#             for _ in range(event_points):
#                 # Time distribution: focused around event with some spread
#                 time_std = event['duration'] / 4
#                 gaze_time = np.random.normal(event_center, time_std)
#                 gaze_time = max(event['start'], min(event['end'], gaze_time))
                
#                 # Channel selection: mostly on event channels, sometimes adjacent
#                 if np.random.random() < 0.8:  # 80%: look at event channels
#                     channel_idx = np.random.choice(channel_indices)
#                 else:  # 20%: look at adjacent channels
#                     base_idx = np.random.choice(channel_indices)
#                     # Add or subtract 1-2 channels
#                     channel_idx = base_idx + np.random.choice([-2, -1, 0, 1, 2])
#                     channel_idx = max(0, min(channel_idx, len(self.labels) - 1))
                
#                 channel_offset = channel_idx * self.offset_step
#                 channel_name = self.labels[channel_idx]
                
#                 # Add variation (less jitter than normal - more focused)
#                 y_variation = np.random.normal(0, self.offset_step / 20)
#                 x_jitter = np.random.normal(0, 0.01)
                
#                 # Abnormal gaze tends to be longer fixations
#                 duration = np.random.uniform(0.3, 0.8)  # Longer for abnormal
                
#                 gaze_point = {
#                     'x': gaze_time + x_jitter,
#                     'y': channel_offset + y_variation,
#                     'time': gaze_time,
#                     'channel': channel_name,
#                     'duration': duration,
#                     'type': 'abnormal'
#                 }
#                 all_gaze_points.append(gaze_point)
        
#         return all_gaze_points

#     def create_heatmap_for_gaze(self, gaze_points, window_height):
#         """Create a heatmap from gaze points with different colors for normal/abnormal"""
#         if not gaze_points:
#             return None
            
#         # Create separate heatmaps for normal and abnormal
#         grid_w, grid_h = 200, 100
#         heatmap_normal = np.zeros((grid_h, grid_w), dtype=float)
#         heatmap_abnormal = np.zeros((grid_h, grid_w), dtype=float)
        
#         for gaze in gaze_points:
#             # Convert time and position to grid coordinates
#             time_normalized = (gaze['x'] - self.current_time) / self.time_window
#             channel_normalized = gaze['y'] / window_height
            
#             x_idx = int(time_normalized * (grid_w - 1))
#             y_idx = int(channel_normalized * (grid_h - 1))
            
#             if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
#                 weight = gaze['duration']
                
#                 if gaze['type'] == 'normal':
#                     heatmap_normal[y_idx, x_idx] += weight
#                 else:
#                     heatmap_abnormal[y_idx, x_idx] += weight
        
#         # Combine heatmaps (abnormal will overlay normal)
#         heatmap_combined = np.zeros((grid_h, grid_w, 3), dtype=float)  # RGB
        
#         # Apply smoothing
#         if np.max(heatmap_normal) > 0:
#             heatmap_normal_smooth = gaussian_filter(heatmap_normal, sigma=self.heatmap_sigma)
#             if np.max(heatmap_normal_smooth) > np.min(heatmap_normal_smooth):
#                 heatmap_norm = (heatmap_normal_smooth - np.min(heatmap_normal_smooth)) / \
#                              (np.max(heatmap_normal_smooth) - np.min(heatmap_normal_smooth))
#             else:
#                 heatmap_norm = heatmap_normal_smooth
            
#             # Blue for normal gaze
#             heatmap_combined[:, :, 2] = heatmap_norm  # Blue channel
        
#         if np.max(heatmap_abnormal) > 0:
#             heatmap_abnormal_smooth = gaussian_filter(heatmap_abnormal, sigma=self.heatmap_sigma)
#             if np.max(heatmap_abnormal_smooth) > np.min(heatmap_abnormal_smooth):
#                 heatmap_abnorm = (heatmap_abnormal_smooth - np.min(heatmap_abnormal_smooth)) / \
#                                (np.max(heatmap_abnormal_smooth) - np.min(heatmap_abnormal_smooth))
#             else:
#                 heatmap_abnorm = heatmap_abnormal_smooth
            
#             # Red for abnormal gaze
#             heatmap_combined[:, :, 0] = heatmap_abnorm  # Red channel
        
#         return heatmap_combined

#     def plot_current_window(self):
#         self.ax.clear()
#         if self.cbar is not None:
#             self.cax.clear()
        
#         # Ensure current_time is within valid range
#         self.current_time = max(0, min(self.current_time, self.duration - self.time_window))
        
#         start_idx = int(self.current_time * self.sample_rate)
#         end_idx = int((self.current_time + self.time_window) * self.sample_rate)
#         end_idx = min(end_idx, len(self.signals[0]))
        
#         # Check if we have valid data to plot
#         if start_idx >= end_idx:
#             print(f"Warning: No data to plot at time {self.current_time}")
#             self.ax.text(0.5, 0.5, "No EEG data in this window", 
#                         transform=self.ax.transAxes, ha='center', va='center', fontsize=12)
#             self.fig.canvas.draw_idle()
#             return

#         print(f"\n=== Plotting window [{self.current_time:.2f}-{self.current_time+self.time_window:.2f}s] ===")

#         # === Plot EEG segment ===
#         channel_offsets = {}
#         window_height = self.offset_step * len(self.labels)
        
#         for i, label in enumerate(self.labels):
#             sig = self.signals[i]
#             segment = sig[start_idx:end_idx]
            
#             # Only plot if we have data
#             if len(segment) > 0:
#                 times = np.linspace(self.current_time, self.current_time + self.time_window, len(segment))
#                 offset = i * self.offset_step
#                 channel_offsets[label] = offset
                
#                 # Color code channels based on whether they're in events
#                 events_in_window = self.get_events_in_current_window()
#                 event_channels = []
#                 for event in events_in_window:
#                     event_channels.extend(event['channels'])
                
#                 if label in event_channels:
#                     line_color = 'red'
#                     line_width = 1.2
#                 else:
#                     line_color = 'black'
#                     line_width = 0.8
                
#                 self.ax.plot(times, segment * self.amplitude_scale + offset, 
#                            color=line_color, linewidth=line_width, zorder=3)
                
#                 # Add channel labels
#                 if len(times) > 0:
#                     self.ax.text(times[0] + 0.05, offset, label,
#                                 va='center', ha='left', fontsize=8, fontweight='bold',
#                                 bbox=dict(facecolor='white', alpha=0.9), zorder=4)

#         # === Get events and normal periods ===
#         events_in_window = self.get_events_in_current_window()
#         window_start = self.current_time
#         window_end = self.current_time + self.time_window
        
#         # Generate gaze points for both normal and abnormal periods
#         normal_periods = self.get_normal_periods_in_window(window_start, window_end, events_in_window)
#         normal_gaze = self.generate_normal_gaze_points(normal_periods, window_height)
#         abnormal_gaze = self.generate_abnormal_gaze_points(events_in_window, window_height)
        
#         all_gaze_points = normal_gaze + abnormal_gaze
        
#         print(f"Events in window: {len(events_in_window)}")
#         print(f"Normal periods: {len(normal_periods)}")
#         print(f"Normal gaze points: {len(normal_gaze)}")
#         print(f"Abnormal gaze points: {len(abnormal_gaze)}")
#         print(f"Total gaze points: {len(all_gaze_points)}")

#         # === Create and plot heatmap ===
#         heatmap = None
#         if all_gaze_points:
#             heatmap = self.create_heatmap_for_gaze(all_gaze_points, window_height)
        
#         if heatmap is not None and np.max(heatmap) > 0:
#             # Define extent for the heatmap
#             extent = [
#                 self.current_time, 
#                 self.current_time + self.time_window, 
#                 0, 
#                 window_height
#             ]
            
#             # Plot the combined heatmap
#             im = self.ax.imshow(
#                 heatmap,
#                 extent=extent,
#                 origin='lower',
#                 alpha=self.heatmap_alpha,
#                 aspect='auto',
#                 zorder=2
#             )
            
#             # Create colorbar with custom colors
#             if self.cbar is None:
#                 self.cbar = self.fig.colorbar(im, cax=self.cax)
#                 self.cbar.set_label('Gaze Intensity (Blue=Normal, Red=Abnormal)', fontsize=9)
#             else:
#                 self.cax.clear()
#                 self.cbar = self.fig.colorbar(im, cax=self.cax)
#                 self.cbar.set_label('Gaze Intensity (Blue=Normal, Red=Abnormal)', fontsize=9)

#         # === Plot individual gaze points (optional - can be commented out) ===
#         if all_gaze_points:
#             # Separate normal and abnormal for coloring
#             normal_x = [g['x'] for g in all_gaze_points if g['type'] == 'normal']
#             normal_y = [g['y'] for g in all_gaze_points if g['type'] == 'normal']
#             abnormal_x = [g['x'] for g in all_gaze_points if g['type'] == 'abnormal']
#             abnormal_y = [g['y'] for g in all_gaze_points if g['type'] == 'abnormal']
            
#             if normal_x:
#                 self.ax.scatter(normal_x, normal_y, c='blue', s=15, alpha=0.5, 
#                                zorder=4, label='Normal Gaze')
#             if abnormal_x:
#                 self.ax.scatter(abnormal_x, abnormal_y, c='red', s=20, alpha=0.7, 
#                                zorder=4, label='Event Gaze')
            
#             self.ax.legend(loc='upper right', fontsize=8)

#         # === Scale bar ===
#         bar_x = self.current_time + self.time_window - 0.5
#         bar_y = -80
#         self.ax.plot([bar_x, bar_x], [bar_y, bar_y + 50], color='black', lw=2, zorder=4)
#         self.ax.plot([bar_x, bar_x + 0.2], [bar_y, bar_y], color='black', lw=2, zorder=4)
#         self.ax.text(bar_x + 0.05, bar_y + 25, "50 µV", va='center', fontsize=8, zorder=4)
#         self.ax.text(bar_x + 0.1, bar_y - 8, "0.2 s", ha='center', fontsize=8, zorder=4)

#         # === Style ===
#         self.ax.set_xlim(self.current_time, self.current_time + self.time_window)
#         self.ax.set_ylim(-100, window_height)
#         self.ax.get_yaxis().set_visible(False)
        
#         # Update title
#         title = f"EEG + Dual Gaze Simulation | Time: {self.current_time:.1f}-{self.current_time+self.time_window:.1f}s"
#         if events_in_window:
#             title += f" | {len(events_in_window)} events"
#         title += f" | {len(normal_gaze)}N/{len(abnormal_gaze)}A gaze points"
        
#         self.ax.set_title(title, fontsize=12, weight='bold')

#         # Hide the colorbar axis if no heatmap
#         if heatmap is None or np.max(heatmap) == 0:
#             self.cax.set_visible(False)
#         else:
#             self.cax.set_visible(True)

#         self.fig.canvas.draw_idle()
#         self.fig.canvas.flush_events()

#     def get_events_in_current_window(self):
#         """Get events that occur within or overlap with the current time window"""
#         window_start = self.current_time
#         window_end = self.current_time + self.time_window
        
#         events_in_window = []
#         for event in self.events:
#             if (event['start'] <= window_end and event['end'] >= window_start):
#                 events_in_window.append(event)
        
#         return events_in_window

#     def scroll_time(self, forward=True):
#         step = self.time_window * (1 - self.overlap_ratio)
#         if forward:
#             new_time = self.current_time + step
#             if new_time < self.duration - self.time_window:
#                 self.current_time = new_time
#             else:
#                 self.current_time = max(0, self.duration - self.time_window)
#         else:
#             new_time = self.current_time - step
#             self.current_time = max(0, new_time)
#         self.plot_current_window()

#     def on_key_press(self, event):
#         if event.key == 'right':
#             self.scroll_time(forward=True)
#         elif event.key == 'left':
#             self.scroll_time(forward=False)
#         elif event.key == 'home':
#             self.current_time = 0.0
#             self.plot_current_window()
#         elif event.key == 'end':
#             self.current_time = max(0, self.duration - self.time_window)
#             self.plot_current_window()
#         elif event.key == 'up':
#             if self.events:
#                 first_event = min(self.events, key=lambda x: x['start'])
#                 self.current_time = max(0, first_event['start'] - self.time_window/2)
#                 self.plot_current_window()
#         elif event.key == 'down':
#             if self.events:
#                 last_event = max(self.events, key=lambda x: x['end'])
#                 self.current_time = max(0, last_event['end'] - self.time_window/2)
#                 self.plot_current_window()

# # Example usage
# if __name__ == "__main__":
#     # Load your events data from CSV
#     events_df = pd.read_csv(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\csv\6.csv")
    
#     # Fill empty demographic cells
#     events_df['Gender'] = events_df['Gender'].fillna(method='ffill')
#     events_df['Age'] = events_df['Age'].fillna(method='ffill') 
#     events_df['File Start'] = events_df['File Start'].fillna(method='ffill')
    
#     EEGViewerWithDualGaze(
#         edf_path=r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\edf\abnormal\0000006.edf",
#         events_data=events_df,
#         time_window=5.0,
#         overlap_ratio=0.2,
#         heatmap_sigma=8,
#         heatmap_alpha=0.7
#     )

#  third varaint man

import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EEGViewerWithDualGaze:
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
        print(f"Loaded {len(self.events)} abnormal events")
        
        if self.events:
            print("\nFirst 5 abnormal events:")
            for i, event in enumerate(self.events[:5]):
                print(f"  Event {i}: {event['start']:.2f}-{event['end']:.2f}s ({event['duration']:.2f}s) on {event['channels']}")

        # ==== Parameters ====
        self.time_window = time_window
        self.overlap_ratio = overlap_ratio
        self.amplitude_scale = amplitude_scale
        self.offset_step = offset_step
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_alpha = heatmap_alpha
        
        # Gaze simulation parameters
        self.GAZE_SAMPLING_RATE = 33  # Hz - typical eye tracker
        self.NORMAL_GAZE_DENSITY = 0.3  # Lower density for normal periods
        self.EVENT_GAZE_DENSITY = 1.0   # Higher density for abnormal periods

        # ==== Matplotlib figure ====
        plt.ion()
        self.fig, (self.ax, self.cax) = plt.subplots(1, 2, figsize=(16, 7), 
                                                     gridspec_kw={'width_ratios': [15, 1]})
        
        # Set white background for both axes
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        self.cax.set_facecolor('white')
        
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
            file_start_seconds = 0
        else:
            file_start_seconds = self.time_string_to_seconds(file_start_time_str)
        
        print(f"File Start: {file_start_time_str} = {file_start_seconds:.2f} seconds")
        
        for _, event in self.events_data.iterrows():
            # Skip rows with missing start times
            if pd.isna(event['Start time']):
                continue
                
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
                'type': 'abnormal',
                'comment': event.get('Comment', '')
            })
        
        events.sort(key=lambda x: x['start'])
        valid_events = [e for e in events if 0 <= e['start'] <= self.duration]
        print(f"Valid events within EDF duration: {len(valid_events)}/{len(events)}")
        
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

    def get_normal_periods_in_window(self, window_start, window_end, events_in_window):
        """Identify normal (non-event) periods within the current window"""
        normal_periods = []
        
        if not events_in_window:
            # No events in window = entire window is normal
            normal_periods.append({
                'start': window_start,
                'end': window_end,
                'duration': window_end - window_start,
                'type': 'normal'
            })
        else:
            # Check gaps between events
            current_time = window_start
            
            for event in sorted(events_in_window, key=lambda x: x['start']):
                if event['start'] > current_time:
                    # Found a normal period before this event
                    normal_periods.append({
                        'start': current_time,
                        'end': event['start'],
                        'duration': event['start'] - current_time,
                        'type': 'normal'
                    })
                current_time = max(current_time, event['end'])
            
            # Check for normal period after last event
            if current_time < window_end:
                normal_periods.append({
                    'start': current_time,
                    'end': window_end,
                    'duration': window_end - current_time,
                    'type': 'normal'
                })
        
        # Filter out very short periods (< 0.1s)
        normal_periods = [p for p in normal_periods if p['duration'] >= 0.1]
        
        return normal_periods

    def generate_normal_gaze_points(self, normal_periods, window_height):
        """Generate gaze points for normal (non-event) periods"""
        all_gaze_points = []
        
        for period in normal_periods:
            # Calculate number of gaze points based on duration and density
            total_points = int(period['duration'] * self.GAZE_SAMPLING_RATE * self.NORMAL_GAZE_DENSITY)
            
            # Ensure minimum points for visualization
            total_points = max(5, min(total_points, 500))
            
            for _ in range(total_points):
                # Random time within normal period
                gaze_time = np.random.uniform(period['start'], period['end'])
                
                # Different gaze patterns for normal periods:
                # 1. Random scanning across all channels
                if np.random.random() < 0.6:  # 60%: random scanning
                    channel_idx = np.random.randint(0, len(self.labels))
                # 2. Focus on central channels (common resting state)
                else:  # 40%: focus on central/sensorimotor channels
                    central_chars = ['C', 'F', 'P']  # Central, Frontal, Parietal
                    central_channels = [i for i, label in enumerate(self.labels) 
                                      if any(label.startswith(char) for char in central_chars)]
                    if central_channels:
                        channel_idx = np.random.choice(central_channels)
                    else:
                        channel_idx = np.random.randint(0, len(self.labels))
                
                channel_offset = channel_idx * self.offset_step
                channel_name = self.labels[channel_idx]
                
                # Add natural variation
                y_variation = np.random.normal(0, self.offset_step / 12)
                x_jitter = np.random.normal(0, 0.03)  # More jitter in normal scanning
                
                # Normal gaze tends to be shorter fixations during scanning
                duration = np.random.uniform(0.1, 0.3)  # Shorter for normal
                
                gaze_point = {
                    'x': gaze_time + x_jitter,
                    'y': channel_offset + y_variation,
                    'time': gaze_time,
                    'channel': channel_name,
                    'duration': duration,
                    'type': 'normal'
                }
                all_gaze_points.append(gaze_point)
        
        return all_gaze_points

    def generate_abnormal_gaze_points(self, events_in_window, window_height):
        """Generate gaze points for abnormal (event) periods"""
        if not events_in_window:
            return []
        
        all_gaze_points = []
        
        for event in events_in_window:
            # Calculate number of gaze points based on duration and density
            event_points = int(event['duration'] * self.GAZE_SAMPLING_RATE * self.EVENT_GAZE_DENSITY)
            event_points = max(10, min(event_points, 1000))
            
            # Event-specific parameters
            event_center = (event['start'] + event['end']) / 2
            event_channels = event['channels']
            
            # Convert channel names to indices
            channel_indices = []
            for ch_name in event_channels:
                if ch_name in self.labels:
                    channel_indices.append(self.labels.index(ch_name))
            
            if not channel_indices:
                continue
            
            for _ in range(event_points):
                # Time distribution: focused around event with some spread
                time_std = event['duration'] / 4
                gaze_time = np.random.normal(event_center, time_std)
                gaze_time = max(event['start'], min(event['end'], gaze_time))
                
                # Channel selection: mostly on event channels, sometimes adjacent
                if np.random.random() < 0.8:  # 80%: look at event channels
                    channel_idx = np.random.choice(channel_indices)
                else:  # 20%: look at adjacent channels
                    base_idx = np.random.choice(channel_indices)
                    # Add or subtract 1-2 channels
                    channel_idx = base_idx + np.random.choice([-2, -1, 0, 1, 2])
                    channel_idx = max(0, min(channel_idx, len(self.labels) - 1))
                
                channel_offset = channel_idx * self.offset_step
                channel_name = self.labels[channel_idx]
                
                # Add variation (less jitter than normal - more focused)
                y_variation = np.random.normal(0, self.offset_step / 20)
                x_jitter = np.random.normal(0, 0.01)
                
                # Abnormal gaze tends to be longer fixations
                duration = np.random.uniform(0.3, 0.8)  # Longer for abnormal
                
                gaze_point = {
                    'x': gaze_time + x_jitter,
                    'y': channel_offset + y_variation,
                    'time': gaze_time,
                    'channel': channel_name,
                    'duration': duration,
                    'type': 'abnormal'
                }
                all_gaze_points.append(gaze_point)
        
        return all_gaze_points

    def create_heatmap_for_gaze(self, gaze_points, window_height):
        """Create a heatmap from gaze points with red color scale"""
        if not gaze_points:
            return None
            
        # Create a single heatmap for all gaze points
        grid_w, grid_h = 200, 100
        heatmap = np.zeros((grid_h, grid_w), dtype=float)
        
        for gaze in gaze_points:
            # Convert time and position to grid coordinates
            time_normalized = (gaze['x'] - self.current_time) / self.time_window
            channel_normalized = gaze['y'] / window_height
            
            x_idx = int(time_normalized * (grid_w - 1))
            y_idx = int(channel_normalized * (grid_h - 1))
            
            if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
                # Different weights for normal vs abnormal
                if gaze['type'] == 'abnormal':
                    weight = gaze['duration'] * 1.5  # Abnormal gets more weight
                else:
                    weight = gaze['duration']
                
                heatmap[y_idx, x_idx] += weight
        
        # Apply smoothing
        if np.max(heatmap) > 0:
            heatmap_smooth = gaussian_filter(heatmap, sigma=self.heatmap_sigma)
            if np.max(heatmap_smooth) > np.min(heatmap_smooth):
                heatmap_norm = (heatmap_smooth - np.min(heatmap_smooth)) / \
                             (np.max(heatmap_smooth) - np.min(heatmap_smooth))
            else:
                heatmap_norm = heatmap_smooth
            
            return heatmap_norm
        
        return None

    def plot_current_window(self):
        self.ax.clear()
        if self.cbar is not None:
            self.cax.clear()
        
        # Set white background for this plot
        self.ax.set_facecolor('white')
        
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
                
                # Color code channels based on whether they're in events
                events_in_window = self.get_events_in_current_window()
                event_channels = []
                for event in events_in_window:
                    event_channels.extend(event['channels'])
                
                # Use dark colors for EEG lines for contrast on white background
                if label in event_channels:
                    line_color = 'darkred'
                    line_width = 1.2
                else:
                    line_color = 'black'
                    line_width = 0.8
                
                self.ax.plot(times, segment * self.amplitude_scale + offset, 
                           color=line_color, linewidth=line_width, zorder=3)
                
                # Add channel labels with white background
                if len(times) > 0:
                    self.ax.text(times[0] + 0.05, offset, label,
                                va='center', ha='left', fontsize=8, fontweight='bold',
                                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, 
                                         boxstyle='round,pad=0.2'), zorder=4)

        # === Get events and normal periods ===
        events_in_window = self.get_events_in_current_window()
        window_start = self.current_time
        window_end = self.current_time + self.time_window
        
        # Generate gaze points for both normal and abnormal periods
        normal_periods = self.get_normal_periods_in_window(window_start, window_end, events_in_window)
        normal_gaze = self.generate_normal_gaze_points(normal_periods, window_height)
        abnormal_gaze = self.generate_abnormal_gaze_points(events_in_window, window_height)
        
        all_gaze_points = normal_gaze + abnormal_gaze
        
        print(f"Events in window: {len(events_in_window)}")
        print(f"Normal periods: {len(normal_periods)}")
        print(f"Normal gaze points: {len(normal_gaze)}")
        print(f"Abnormal gaze points: {len(abnormal_gaze)}")
        print(f"Total gaze points: {len(all_gaze_points)}")

        # === Create and plot heatmap ===
        heatmap = None
        if all_gaze_points:
            heatmap = self.create_heatmap_for_gaze(all_gaze_points, window_height)
        
        if heatmap is not None and np.max(heatmap) > 0:
            # Define extent for the heatmap
            extent = [
                self.current_time, 
                self.current_time + self.time_window, 
                0, 
                window_height
            ]
            
            # Use red colormap for heatmap (single color)
            # 'Reds' goes from white to dark red
            im = self.ax.imshow(
                heatmap,
                extent=extent,
                origin='lower',
                cmap='Reds',  # Single color red scale
                alpha=self.heatmap_alpha,
                aspect='auto',
                zorder=2,
                vmin=0,  # Ensure minimum is 0 (white)
                vmax=1   # Ensure maximum is 1 (dark red)
            )
            
            # Create colorbar with red scale
            if self.cbar is None:
                self.cbar = self.fig.colorbar(im, cax=self.cax)
                self.cbar.set_label('Gaze Intensity', fontsize=9)
                self.cbar.ax.set_facecolor('white')
            else:
                self.cax.clear()
                self.cbar = self.fig.colorbar(im, cax=self.cax)
                self.cbar.set_label('Gaze Intensity', fontsize=9)
                self.cbar.ax.set_facecolor('white')

        # === Plot individual gaze points (optional - can be commented out) ===
        if all_gaze_points:
            # Plot all gaze points in red
            gaze_x = [g['x'] for g in all_gaze_points]
            gaze_y = [g['y'] for g in all_gaze_points]
            
            # Use different marker sizes for normal vs abnormal
            sizes = []
            colors = []
            for g in all_gaze_points:
                if g['type'] == 'abnormal':
                    sizes.append(20)
                    colors.append('darkred')
                else:
                    sizes.append(10)
                    colors.append('red')
            
            self.ax.scatter(gaze_x, gaze_y, c=colors, s=sizes, alpha=0.7, 
                           zorder=4, edgecolors='white', linewidth=0.5)

        # === Scale bar ===
        bar_x = self.current_time + self.time_window - 0.5
        bar_y = -80
        self.ax.plot([bar_x, bar_x], [bar_y, bar_y + 50], color='black', lw=2, zorder=4)
        self.ax.plot([bar_x, bar_x + 0.2], [bar_y, bar_y], color='black', lw=2, zorder=4)
        self.ax.text(bar_x + 0.05, bar_y + 25, "50 µV", va='center', fontsize=8, 
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.8), zorder=4)
        self.ax.text(bar_x + 0.1, bar_y - 8, "0.2 s", ha='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.8), zorder=4)

        # === Style ===
        self.ax.set_xlim(self.current_time, self.current_time + self.time_window)
        self.ax.set_ylim(-100, window_height)
        self.ax.get_yaxis().set_visible(False)
        
        # Set axis colors for white background
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['top'].set_color('black')
        self.ax.spines['right'].set_color('black')
        self.ax.spines['left'].set_color('black')
        
        # Update title
        title = f"EEG + Gaze Intensity Heatmap | Time: {self.current_time:.1f}-{self.current_time+self.time_window:.1f}s"
        if events_in_window:
            title += f" | {len(events_in_window)} events"
        title += f" | {len(normal_gaze)}N/{len(abnormal_gaze)}A gaze points"
        
        self.ax.set_title(title, fontsize=12, weight='bold', color='black')

        # Hide the colorbar axis if no heatmap
        if heatmap is None or np.max(heatmap) == 0:
            self.cax.set_visible(False)
        else:
            self.cax.set_visible(True)
            self.cax.set_facecolor('white')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def get_events_in_current_window(self):
        """Get events that occur within or overlap with the current time window"""
        window_start = self.current_time
        window_end = self.current_time + self.time_window
        
        events_in_window = []
        for event in self.events:
            if (event['start'] <= window_end and event['end'] >= window_start):
                events_in_window.append(event)
        
        return events_in_window

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
            if self.events:
                first_event = min(self.events, key=lambda x: x['start'])
                self.current_time = max(0, first_event['start'] - self.time_window/2)
                self.plot_current_window()
        elif event.key == 'down':
            if self.events:
                last_event = max(self.events, key=lambda x: x['end'])
                self.current_time = max(0, last_event['end'] - self.time_window/2)
                self.plot_current_window()

# Example usage
if __name__ == "__main__":
    # Load your events data from CSV
    events_df = pd.read_csv(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\csv\6.csv")
    
    # Fill empty demographic cells
    events_df['Gender'] = events_df['Gender'].fillna(method='ffill')
    events_df['Age'] = events_df['Age'].fillna(method='ffill') 
    events_df['File Start'] = events_df['File Start'].fillna(method='ffill')
    
    EEGViewerWithDualGaze(
        edf_path=r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\edf\abnormal\0000006.edf",
        events_data=events_df,
        time_window=5.0,
        overlap_ratio=0.2,
        heatmap_sigma=8,
        heatmap_alpha=0.7
    )

