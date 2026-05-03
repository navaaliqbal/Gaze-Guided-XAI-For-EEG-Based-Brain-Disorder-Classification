# import pyedflib
# import numpy as np
# import json
# import pandas as pd
# import random
# from datetime import datetime, timedelta
# import os

# class EEGGazeSimulator:
#     def __init__(self, edf_path, events_data):
#         # ==== Load EDF ====
#         self.f = pyedflib.EdfReader(edf_path)
#         self.n_signals = self.f.signals_in_file
#         self.labels = self.f.getSignalLabels()
#         self.sample_rate = int(self.f.getSampleFrequency(0))
#         self.signals = [self.f.readSignal(i) for i in range(self.n_signals)]
#         self.duration = len(self.signals[0]) / self.sample_rate
#         self.f.close()
        
#         # Store edf filename
#         self.edf_filename = os.path.basename(edf_path)

#         print(f"=== EEG INFO ===")
#         print(f"EEG Duration: {self.duration:.2f} seconds")
#         print(f"EEG Sample Rate: {self.sample_rate} Hz")
#         print(f"Number of channels: {self.n_signals}")
#         print(f"EDF File: {self.edf_filename}")

#         # ==== Process Events Data ====
#         self.events_data = events_data
#         self.events = self.process_events_data()
        
#         print(f"\n=== EVENTS INFO ===")
#         print(f"Loaded {len(self.events)} abnormal events")
        
#         if self.events:
#             print("\nFirst 5 abnormal events:")
#             for i, event in enumerate(self.events[:5]):
#                 print(f"  Event {i}: {event['start']:.2f}-{event['end']:.2f}s ({event['duration']:.2f}s) on {event['channels']}")

#         # ==== Gaze simulation parameters ====
#         self.GAZE_SAMPLING_RATE = 33  # Hz - typical eye tracker
#         self.NORMAL_GAZE_DENSITY = 0.3  # Lower density for normal periods
#         self.EVENT_GAZE_DENSITY = 1.0   # Higher density for abnormal periods
        
#         # Visualization parameters (for coordinate generation)
#         self.offset_step = 200  # Vertical spacing between channels
        
#         # Generate gaze data for the entire EEG file
#         self.gaze_data = self.generate_complete_gaze_data()

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

#     def get_normal_periods(self, events):
#         """Identify normal (non-event) periods in the entire EEG"""
#         normal_periods = []
        
#         if not events:
#             # No events = entire file is normal
#             normal_periods.append({
#                 'start': 0,
#                 'end': self.duration,
#                 'duration': self.duration,
#                 'type': 'normal'
#             })
#         else:
#             # Check gaps between events
#             current_time = 0
            
#             for event in sorted(events, key=lambda x: x['start']):
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
#             if current_time < self.duration:
#                 normal_periods.append({
#                     'start': current_time,
#                     'end': self.duration,
#                     'duration': self.duration - current_time,
#                     'type': 'normal'
#                 })
        
#         # Filter out very short periods (< 0.1s)
#         normal_periods = [p for p in normal_periods if p['duration'] >= 0.1]
        
#         return normal_periods

#     def generate_normal_gaze_points(self, normal_periods):
#         """Generate gaze points for normal (non-event) periods"""
#         all_gaze_points = []
        
#         # Calculate total points for all normal periods
#         total_normal_duration = sum(p['duration'] for p in normal_periods)
#         total_points = int(total_normal_duration * self.GAZE_SAMPLING_RATE * self.NORMAL_GAZE_DENSITY)
        
#         # Distribute points proportionally across periods
#         for period in normal_periods:
#             period_fraction = period['duration'] / total_normal_duration
#             points_for_period = max(1, int(total_points * period_fraction))
            
#             for _ in range(points_for_period):
#                 # Random time within this period
#                 gaze_time = np.random.uniform(period['start'], period['end'])
                
#                 # Determine which channel to look at
#                 # 60%: Random scanning across all channels
#                 if np.random.random() < 0.6:
#                     channel_idx = np.random.randint(0, len(self.labels))
#                 # 40%: Focus on central/sensorimotor channels
#                 else:
#                     central_chars = ['C', 'F', 'P']
#                     central_channels = [i for i, label in enumerate(self.labels) 
#                                       if any(label.startswith(char) for char in central_chars)]
#                     if central_channels:
#                         channel_idx = np.random.choice(central_channels)
#                     else:
#                         channel_idx = np.random.randint(0, len(self.labels))
                
#                 channel_name = self.labels[channel_idx]
#                 channel_offset = channel_idx * self.offset_step
                
#                 # Add natural variation
#                 y_variation = np.random.normal(0, self.offset_step / 12)
#                 x_jitter = np.random.normal(0, 0.03)
                
#                 # Get EEG value at this time
#                 eeg_value = self.get_eeg_value(channel_name, gaze_time)
                
#                 # Generate gaze point
#                 gaze_point = self.create_gaze_point(
#                     timestamp=gaze_time,
#                     eeg_time=gaze_time,
#                     channel_name=channel_name,
#                     eeg_value=eeg_value,
#                     channel_offset=channel_offset,
#                     x_jitter=x_jitter,
#                     y_variation=y_variation,
#                     gaze_type='normal'
#                 )
#                 all_gaze_points.append(gaze_point)
        
#         return all_gaze_points

#     def generate_abnormal_gaze_points(self, events):
#         """Generate gaze points for abnormal (event) periods"""
#         all_gaze_points = []
        
#         for event in events:
#             # Calculate number of gaze points for this event
#             event_points = int(event['duration'] * self.GAZE_SAMPLING_RATE * self.EVENT_GAZE_DENSITY)
#             event_points = max(10, min(event_points, 1000))
            
#             # Get channel indices for this event
#             channel_indices = []
#             for ch_name in event['channels']:
#                 if ch_name in self.labels:
#                     channel_indices.append(self.labels.index(ch_name))
            
#             if not channel_indices:
#                 continue
            
#             for _ in range(event_points):
#                 # Time distribution: focused around event center
#                 event_center = (event['start'] + event['end']) / 2
#                 time_std = event['duration'] / 4
#                 gaze_time = np.random.normal(event_center, time_std)
#                 gaze_time = max(event['start'], min(event['end'], gaze_time))
                
#                 # Channel selection
#                 if np.random.random() < 0.8:  # 80%: look at event channels
#                     channel_idx = np.random.choice(channel_indices)
#                 else:  # 20%: look at adjacent channels
#                     base_idx = np.random.choice(channel_indices)
#                     channel_idx = base_idx + np.random.choice([-2, -1, 0, 1, 2])
#                     channel_idx = max(0, min(channel_idx, len(self.labels) - 1))
                
#                 channel_name = self.labels[channel_idx]
#                 channel_offset = channel_idx * self.offset_step
                
#                 # Add variation (less jitter than normal - more focused)
#                 y_variation = np.random.normal(0, self.offset_step / 20)
#                 x_jitter = np.random.normal(0, 0.01)
                
#                 # Get EEG value at this time
#                 eeg_value = self.get_eeg_value(channel_name, gaze_time)
                
#                 # Generate gaze point
#                 gaze_point = self.create_gaze_point(
#                     timestamp=gaze_time,
#                     eeg_time=gaze_time,
#                     channel_name=channel_name,
#                     eeg_value=eeg_value,
#                     channel_offset=channel_offset,
#                     x_jitter=x_jitter,
#                     y_variation=y_variation,
#                     gaze_type='abnormal'
#                 )
#                 all_gaze_points.append(gaze_point)
        
#         return all_gaze_points

#     def get_eeg_value(self, channel_name, time):
#         """Get EEG value at specific time for specific channel"""
#         try:
#             channel_idx = self.labels.index(channel_name)
#             sample_idx = int(time * self.sample_rate)
            
#             if 0 <= sample_idx < len(self.signals[channel_idx]):
#                 return float(self.signals[channel_idx][sample_idx])
#         except Exception as e:
#             print(f"Warning: Could not get EEG value for {channel_name} at {time}s: {e}")
        
#         return None

#     def create_gaze_point(self, timestamp, eeg_time, channel_name, eeg_value, 
#                          channel_offset, x_jitter, y_variation, gaze_type):
#         """Create a single gaze point dictionary in the desired format"""
        
#         # Create a realistic timestamp (starting from current time minus duration)
#         base_timestamp = datetime.now().timestamp() - self.duration
#         absolute_timestamp = base_timestamp + timestamp
        
#         # Calculate coordinates
#         x_coord = eeg_time + x_jitter
#         y_coord = channel_offset + y_variation
        
#         # Simulate raw screen coordinates (assuming 1920x1080 display)
#         raw_x = int(((x_coord / self.duration) * 1600) + 160)  # Scale to screen
#         raw_y = int(1080 - (y_coord / (len(self.labels) * self.offset_step)) * 800) - 140
        
#         return {
#             "timestamp": absolute_timestamp,
#             "time": float(x_coord),  # EEG time coordinate
#             "channel": channel_name,
#             "value": eeg_value,
#             "coords": {
#                 "x": float(x_coord),
#                 "y": float(y_coord)
#             },
#             "raw": {
#                 "x": raw_x,
#                 "y": raw_y
#             },
#             "gaze_type": gaze_type,
#             "metadata": {
#                 "sampling_rate": self.GAZE_SAMPLING_RATE,
#                 "eeg_sample_rate": self.sample_rate,
#                 "channel_index": self.labels.index(channel_name) if channel_name in self.labels else -1
#             }
#         }

#     def generate_complete_gaze_data(self):
#         """Generate gaze data for the entire EEG file"""
#         print("\n=== Generating Complete Gaze Data ===")
        
#         # Get all normal and abnormal periods
#         normal_periods = self.get_normal_periods(self.events)
        
#         print(f"Normal periods: {len(normal_periods)}")
#         print(f"Abnormal events: {len(self.events)}")
        
#         # Generate gaze points
#         normal_gaze = self.generate_normal_gaze_points(normal_periods)
#         abnormal_gaze = self.generate_abnormal_gaze_points(self.events)
        
#         all_gaze = normal_gaze + abnormal_gaze
        
#         # Sort by timestamp
#         all_gaze.sort(key=lambda x: x['timestamp'])
        
#         print(f"Generated {len(normal_gaze)} normal gaze points")
#         print(f"Generated {len(abnormal_gaze)} abnormal gaze points")
#         print(f"Total gaze points: {len(all_gaze)}")
        
#         return all_gaze

#     def create_json_output(self, output_path=None, time_window=10.0):
#         """Create JSON output in the desired format"""
#         if output_path is None:
#             # Create default output path
#             base_name = os.path.splitext(self.edf_filename)[0]
#             output_path = f"simulated_gaze_{base_name}.json"
        
#         # Calculate session start and end times
#         session_start = min(gaze['timestamp'] for gaze in self.gaze_data)
#         session_end = max(gaze['timestamp'] for gaze in self.gaze_data)
        
#         # Create the JSON structure
#         output_data = {
#             "edf_file": self.edf_filename,
#             "session": [
#                 {
#                     "time_window": float(time_window),
#                     "time_window_start": 0.0,
#                     "start_time": float(session_start),
#                     "end_time": float(session_end),
#                     "gaze_data": self.gaze_data
#                 }
#             ]
#         }
        
#         # Save to file
#         with open(output_path, 'w') as f:
#             json.dump(output_data, f, indent=2)
        
#         print(f"\n=== JSON Output Saved ===")
#         print(f"File: {output_path}")
#         print(f"Session duration: {session_end - session_start:.2f} seconds")
#         print(f"Total gaze points: {len(self.gaze_data)}")
        
#         return output_path

#     def get_statistics(self):
#         """Get statistics about the generated gaze data"""
#         stats = {
#             "total_points": len(self.gaze_data),
#             "normal_points": sum(1 for g in self.gaze_data if g['gaze_type'] == 'normal'),
#             "abnormal_points": sum(1 for g in self.gaze_data if g['gaze_type'] == 'abnormal'),
#             "unique_channels": len(set(g['channel'] for g in self.gaze_data if g['channel'])),
#             "time_range": {
#                 "start": min(g['timestamp'] for g in self.gaze_data),
#                 "end": max(g['timestamp'] for g in self.gaze_data),
#                 "duration": max(g['timestamp'] for g in self.gaze_data) - min(g['timestamp'] for g in self.gaze_data)
#             }
#         }
        
#         # Calculate gaze point density
#         stats['gaze_density_hz'] = stats['total_points'] / stats['time_range']['duration']
        
#         return stats

# # Example usage
# if __name__ == "__main__":
#     # Load your events data from CSV
#     events_df = pd.read_csv(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\csv\6.csv")
    
#     # Fill empty demographic cells
#     events_df['Gender'] = events_df['Gender'].fillna(method='ffill')
#     events_df['Age'] = events_df['Age'].fillna(method='ffill') 
#     events_df['File Start'] = events_df['File Start'].fillna(method='ffill')
    
#     # Create simulator
#     simulator = EEGGazeSimulator(
#         edf_path=r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\edf\abnormal\0000006.edf",
#         events_data=events_df
#     )
    
#     # Create JSON output
#     output_file = simulator.create_json_output("D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/simulation_gazepoints/0000006.json",time_window=5.0)
#     # Get statistics
#     stats = simulator.get_statistics()
#     print("\n=== Statistics ===")
#     for key, value in stats.items():
#         if isinstance(value, dict):
#             print(f"{key}:")
#             for sub_key, sub_value in value.items():
#                 print(f"  {sub_key}: {sub_value}")
#         else:
#             print(f"{key}: {value}")
    
#     # Print sample gaze points
#     print("\n=== Sample Gaze Points (first 5) ===")
#     for i, gaze in enumerate(simulator.gaze_data[:5]):
#         print(f"\nGaze Point {i}:")
#         print(f"  Timestamp: {gaze['timestamp']:.3f}")
#         print(f"  Time (EEG): {gaze['time']:.3f}s")
#         print(f"  Channel: {gaze['channel']}")
#         print(f"  EEG Value: {gaze['value']}")
#         print(f"  Coords: ({gaze['coords']['x']:.1f}, {gaze['coords']['y']:.1f})")
#         print(f"  Raw: ({gaze['raw']['x']}, {gaze['raw']['y']})")
#         print(f"  Type: {gaze['gaze_type']}")



# # import pyedflib
# # import numpy as np
# # import json
# # import pandas as pd
# # import random
# # from datetime import datetime, timedelta
# # import os

# # class EEGGazeSimulatorClustered:
# #     def __init__(self, edf_path, events_data, cluster_size=50, fixation_duration=0.3):
# #         # ==== Load EDF ====
# #         self.f = pyedflib.EdfReader(edf_path)
# #         self.n_signals = self.f.signals_in_file
# #         self.labels = self.f.getSignalLabels()
# #         self.sample_rate = int(self.f.getSampleFrequency(0))
# #         self.signals = [self.f.readSignal(i) for i in range(self.n_signals)]
# #         self.duration = len(self.signals[0]) / self.sample_rate
# #         self.f.close()
        
# #         # Store edf filename
# #         self.edf_filename = os.path.basename(edf_path)

# #         print(f"=== EEG INFO ===")
# #         print(f"EEG Duration: {self.duration:.2f} seconds")
# #         print(f"EEG Sample Rate: {self.sample_rate} Hz")
# #         print(f"Number of channels: {self.n_signals}")
# #         print(f"EDF File: {self.edf_filename}")

# #         # ==== Process Events Data ====
# #         self.events_data = events_data
# #         self.events = self.process_events_data()
        
# #         print(f"\n=== EVENTS INFO ===")
# #         print(f"Loaded {len(self.events)} abnormal events")
        
# #         if self.events:
# #             print("\nFirst 5 abnormal events:")
# #             for i, event in enumerate(self.events[:5]):
# #                 print(f"  Event {i}: {event['start']:.2f}-{event['end']:.2f}s ({event['duration']:.2f}s) on {event['channels']}")

# #         # ==== NEW: Clustering parameters ====
# #         self.CLUSTER_SIZE = cluster_size  # How many pixels to spread within a cluster (like words)
# #         self.FIXATION_DURATION = fixation_duration  # Average fixation duration
# #         self.GAZE_SAMPLING_RATE = 33  # Hz - typical eye tracker
        
# #         # For realistic simulation: channel positions are now CLUSTERED
# #         # Instead of spreading across 0-4000px, we'll create clusters around each channel
# #         self.channel_positions = self.create_channel_clusters()
        
# #         # Generate gaze data for the entire EEG file
# #         self.gaze_data = self.generate_clustered_gaze_data()

# #     def create_channel_clusters(self):
# #         """Create clustered positions for each channel (like words on a page)"""
# #         positions = {}
        
# #         # For real words: positions are in readable range (e.g., 100-1000 pixels)
# #         # For EEG: we'll create channel clusters at reasonable positions
# #         base_y = 300  # Start at 300px (not 0)
# #         channel_spacing = 120  # 120px between channels (like lines of text)
        
# #         for i, label in enumerate(self.labels):
# #             # Each channel gets a "cluster area" where gaze points will be clustered
# #             center_x = 400 + (i % 3) * 300  # Stagger clusters horizontally
# #             center_y = base_y + i * channel_spacing
            
# #             positions[label] = {
# #                 'center_x': center_x,
# #                 'center_y': center_y,
# #                 'cluster_radius': self.CLUSTER_SIZE  # How tightly points cluster
# #             }
        
# #         print(f"\nCreated {len(positions)} channel clusters")
# #         print(f"Cluster radius: {self.CLUSTER_SIZE}px")
# #         return positions

# #     def process_events_data(self):
# #         """Process events data and map to EDF timeline using File Start as reference"""
# #         events = []
        
# #         # Get the file start time from the first row
# #         file_start_time_str = None
# #         for _, event in self.events_data.iterrows():
# #             if pd.notna(event['File Start']):
# #                 file_start_time_str = event['File Start']
# #                 break
        
# #         if file_start_time_str is None:
# #             print("Warning: Could not find File Start time in events data")
# #             file_start_seconds = 0
# #         else:
# #             file_start_seconds = self.time_string_to_seconds(file_start_time_str)
        
# #         print(f"File Start: {file_start_time_str} = {file_start_seconds:.2f} seconds")
        
# #         for _, event in self.events_data.iterrows():
# #             # Skip rows with missing start times
# #             if pd.isna(event['Start time']):
# #                 continue
                
# #             start_sec = self.time_string_to_seconds(event['Start time'])
# #             end_sec = self.time_string_to_seconds(event['End time'])
            
# #             # Convert to relative time (File Start = 0 seconds)
# #             start_relative = start_sec - file_start_seconds
# #             end_relative = end_sec - file_start_seconds
            
# #             # Parse channel names
# #             channels = event['Channel names'].split()
            
# #             events.append({
# #                 'start': start_relative,
# #                 'end': end_relative,
# #                 'channels': channels,
# #                 'duration': end_relative - start_relative,
# #                 'type': 'abnormal',
# #                 'comment': event.get('Comment', '')
# #             })
        
# #         events.sort(key=lambda x: x['start'])
# #         valid_events = [e for e in events if 0 <= e['start'] <= self.duration]
# #         print(f"Valid events within EDF duration: {len(valid_events)}/{len(events)}")
        
# #         return valid_events
    
# #     def time_string_to_seconds(self, time_str):
# #         """Convert time string to total seconds"""
# #         try:
# #             if ':' in time_str:
# #                 parts = time_str.split(':')
# #                 if len(parts) == 4:  # HH:MM:SS:ms
# #                     hours, minutes, seconds, milliseconds = parts
# #                     total_seconds = (int(hours) * 3600 + 
# #                                    int(minutes) * 60 + 
# #                                    int(seconds) + 
# #                                    int(milliseconds) / 1000)
# #                 elif len(parts) == 3:  # HH:MM:SS
# #                     hours, minutes, seconds = parts
# #                     total_seconds = (int(hours) * 3600 + 
# #                                    int(minutes) * 60 + 
# #                                    float(seconds))
# #                 else:
# #                     total_seconds = float(time_str)
# #             else:
# #                 total_seconds = float(time_str)
                
# #             return total_seconds
# #         except Exception as e:
# #             print(f"Error parsing time string '{time_str}': {e}")
# #             return 0.0

# #     def get_normal_periods(self, events):
# #         """Identify normal (non-event) periods in the entire EEG"""
# #         normal_periods = []
        
# #         if not events:
# #             # No events = entire file is normal
# #             normal_periods.append({
# #                 'start': 0,
# #                 'end': self.duration,
# #                 'duration': self.duration,
# #                 'type': 'normal'
# #             })
# #         else:
# #             # Check gaps between events
# #             current_time = 0
            
# #             for event in sorted(events, key=lambda x: x['start']):
# #                 if event['start'] > current_time:
# #                     # Found a normal period before this event
# #                     normal_periods.append({
# #                         'start': current_time,
# #                         'end': event['start'],
# #                         'duration': event['start'] - current_time,
# #                         'type': 'normal'
# #                     })
# #                 current_time = max(current_time, event['end'])
            
# #             # Check for normal period after last event
# #             if current_time < self.duration:
# #                 normal_periods.append({
# #                     'start': current_time,
# #                     'end': self.duration,
# #                     'duration': self.duration - current_time,
# #                     'type': 'normal'
# #                 })
        
# #         # Filter out very short periods (< 0.5s)
# #         normal_periods = [p for p in normal_periods if p['duration'] >= 0.5]
        
# #         return normal_periods

# #     def generate_normal_gaze_clusters(self, normal_periods):
# #         """Generate CLUSTERED gaze points for normal periods"""
# #         all_gaze_points = []
        
# #         for period in normal_periods:
# #             # For normal periods: simulate casual reading/scanning
# #             # Create 2-4 "fixation clusters" per second of normal period
# #             num_clusters = max(1, int(period['duration'] * 3))
            
# #             for cluster_num in range(num_clusters):
# #                 # Random time within this period for this cluster
# #                 cluster_time = np.random.uniform(period['start'], period['end'])
                
# #                 # Choose a random channel to "read"
# #                 channel_idx = np.random.randint(0, len(self.labels))
# #                 channel_name = self.labels[channel_idx]
# #                 cluster_info = self.channel_positions[channel_name]
                
# #                 # Create a cluster of gaze points (like reading a word)
# #                 # Number of points in cluster = duration * sampling rate
# #                 cluster_duration = np.random.uniform(0.2, 0.6)  # Fixation duration
# #                 points_in_cluster = max(2, int(cluster_duration * self.GAZE_SAMPLING_RATE))
                
# #                 for point_num in range(points_in_cluster):
# #                     # Time within this fixation cluster
# #                     point_time = cluster_time + (point_num * 0.03)  # 30ms between points
                    
# #                     # Coordinates CLUSTERED around channel position
# #                     x = cluster_info['center_x'] + np.random.normal(0, self.CLUSTER_SIZE/3)
# #                     y = cluster_info['center_y'] + np.random.normal(0, self.CLUSTER_SIZE/3)
                    
# #                     # Get EEG value
# #                     eeg_value = self.get_eeg_value(channel_name, point_time)
                    
# #                     # Create gaze point
# #                     gaze_point = self.create_clustered_gaze_point(
# #                         eeg_time=point_time,
# #                         screen_x=x,
# #                         screen_y=y,
# #                         channel_name=channel_name,
# #                         eeg_value=eeg_value,
# #                         gaze_type='normal',
# #                         cluster_id=cluster_num
# #                     )
# #                     all_gaze_points.append(gaze_point)
        
# #         return all_gaze_points

# #     def generate_abnormal_gaze_clusters(self, events):
# #         """Generate CLUSTERED gaze points for abnormal events"""
# #         all_gaze_points = []
        
# #         for event_idx, event in enumerate(events):
# #             # For abnormal events: more INTENSE clustering on event channels
# #             event_channels = [ch for ch in event['channels'] if ch in self.labels]
            
# #             if not event_channels:
# #                 continue
            
# #             # Create multiple fixation clusters during this event
# #             # More clusters for longer events
# #             num_clusters = max(1, int(event['duration'] * 4))  # 4 clusters per second
            
# #             for cluster_num in range(num_clusters):
# #                 # Time within event (biased toward center)
# #                 event_center = (event['start'] + event['end']) / 2
# #                 time_std = event['duration'] / 3
# #                 cluster_time = np.random.normal(event_center, time_std)
# #                 cluster_time = max(event['start'], min(event['end'], cluster_time))
                
# #                 # Choose an event channel (focus on event channels)
# #                 if np.random.random() < 0.8:  # 80% on event channels
# #                     channel_name = random.choice(event_channels)
# #                 else:  # 20% on any channel
# #                     channel_name = random.choice(self.labels)
                
# #                 cluster_info = self.channel_positions[channel_name]
                
# #                 # Create TIGHTER cluster for abnormal events (more focused)
# #                 cluster_duration = np.random.uniform(0.3, 0.8)  # Longer fixations
# #                 points_in_cluster = max(3, int(cluster_duration * self.GAZE_SAMPLING_RATE))
                
# #                 for point_num in range(points_in_cluster):
# #                     # Time within this fixation cluster
# #                     point_time = cluster_time + (point_num * 0.03)
                    
# #                     # TIGHTER clustering for abnormal (more focused)
# #                     x = cluster_info['center_x'] + np.random.normal(0, self.CLUSTER_SIZE/4)
# #                     y = cluster_info['center_y'] + np.random.normal(0, self.CLUSTER_SIZE/4)
                    
# #                     # Get EEG value
# #                     eeg_value = self.get_eeg_value(channel_name, point_time)
                    
# #                     # Create gaze point
# #                     gaze_point = self.create_clustered_gaze_point(
# #                         eeg_time=point_time,
# #                         screen_x=x,
# #                         screen_y=y,
# #                         channel_name=channel_name,
# #                         eeg_value=eeg_value,
# #                         gaze_type='abnormal',
# #                         cluster_id=f"event{event_idx}_cluster{cluster_num}"
# #                     )
# #                     all_gaze_points.append(gaze_point)
        
# #         return all_gaze_points

# #     def get_eeg_value(self, channel_name, time):
# #         """Get EEG value at specific time for specific channel"""
# #         try:
# #             channel_idx = self.labels.index(channel_name)
# #             sample_idx = int(time * self.sample_rate)
            
# #             if 0 <= sample_idx < len(self.signals[channel_idx]):
# #                 return float(self.signals[channel_idx][sample_idx])
# #         except Exception as e:
# #             pass
        
# #         return None

# #     def create_clustered_gaze_point(self, eeg_time, screen_x, screen_y, 
# #                                    channel_name, eeg_value, gaze_type, cluster_id):
# #         """Create a gaze point with CLUSTERED coordinates"""
        
# #         # Create realistic timestamp
# #         base_timestamp = datetime.now().timestamp() - self.duration
# #         absolute_timestamp = base_timestamp + eeg_time
        
# #         # Ensure coordinates are in reasonable range (like real screen)
# #         screen_x = max(100, min(screen_x, 1800))  # Keep within 100-1800px
# #         screen_y = max(100, min(screen_y, 1000))  # Keep within 100-1000px
        
# #         # For EEG plot coordinates: x = eeg_time, y based on channel position
# #         eeg_y = self.channel_positions[channel_name]['center_y']
        
# #         return {
# #             "timestamp": absolute_timestamp,
# #             "time": float(eeg_time),  # EEG time
# #             "channel": channel_name,
# #             "value": eeg_value,
# #             "coords": {
# #                 "x": float(eeg_time),  # EEG time as x-coordinate
# #                 "y": float(eeg_y)      # Channel position as y-coordinate
# #             },
# #             "raw": {
# #                 "x": int(screen_x),    # Screen pixel X (clustered)
# #                 "y": int(screen_y)     # Screen pixel Y (clustered)
# #             },
# #             "gaze_type": gaze_type,
# #             "cluster_id": str(cluster_id),
# #             "metadata": {
# #                 "sampling_rate": self.GAZE_SAMPLING_RATE,
# #                 "eeg_sample_rate": self.sample_rate,
# #                 "channel_index": self.labels.index(channel_name) if channel_name in self.labels else -1,
# #                 "cluster_radius": self.CLUSTER_SIZE
# #             }
# #         }

# #     def generate_clustered_gaze_data(self):
# #         """Generate gaze data with REALISTIC CLUSTERING"""
# #         print("\n=== Generating CLUSTERED Gaze Data ===")
        
# #         # Get all normal and abnormal periods
# #         normal_periods = self.get_normal_periods(self.events)
        
# #         print(f"Normal periods: {len(normal_periods)}")
# #         print(f"Abnormal events: {len(self.events)}")
# #         print(f"Cluster size: {self.CLUSTER_SIZE}px")
# #         print(f"Target fixation duration: {self.FIXATION_DURATION}s")
        
# #         # Generate clustered gaze points
# #         normal_gaze = self.generate_normal_gaze_clusters(normal_periods)
# #         abnormal_gaze = self.generate_abnormal_gaze_clusters(self.events)
        
# #         all_gaze = normal_gaze + abnormal_gaze
        
# #         # Sort by timestamp
# #         all_gaze.sort(key=lambda x: x['timestamp'])
        
# #         print(f"\n=== Generation Complete ===")
# #         print(f"Normal gaze points: {len(normal_gaze)}")
# #         print(f"Abnormal gaze points: {len(abnormal_gaze)}")
# #         print(f"Total gaze points: {len(all_gaze)}")
        
# #         # Print coordinate ranges
# #         xs = [p['raw']['x'] for p in all_gaze]
# #         ys = [p['raw']['y'] for p in all_gaze]
# #         print(f"\nCoordinate Ranges:")
# #         print(f"  Raw X: {min(xs)} - {max(xs)} px")
# #         print(f"  Raw Y: {min(ys)} - {max(ys)} px")
# #         print(f"  EEG X (time): {min(p['time'] for p in all_gaze):.1f} - {max(p['time'] for p in all_gaze):.1f} s")
        
# #         return all_gaze

# #     def create_json_output(self, output_path=None, time_window=10.0):
# #         """Create JSON output in the desired format"""
# #         if output_path is None:
# #             # Create default output path
# #             base_name = os.path.splitext(self.edf_filename)[0]
# #             output_path = f"simulated_gaze_clustered_{base_name}.json"
        
# #         # Calculate session start and end times
# #         session_start = min(gaze['timestamp'] for gaze in self.gaze_data)
# #         session_end = max(gaze['timestamp'] for gaze in self.gaze_data)
        
# #         # Create the JSON structure
# #         output_data = {
# #             "edf_file": self.edf_filename,
# #             "cluster_parameters": {
# #                 "cluster_size_px": self.CLUSTER_SIZE,
# #                 "fixation_duration_s": self.FIXATION_DURATION,
# #                 "sampling_rate_hz": self.GAZE_SAMPLING_RATE
# #             },
# #             "session": [
# #                 {
# #                     "time_window": float(time_window),
# #                     "time_window_start": 0.0,
# #                     "start_time": float(session_start),
# #                     "end_time": float(session_end),
# #                     "gaze_data": self.gaze_data
# #                 }
# #             ]
# #         }
        
# #         # Save to file
# #         with open(output_path, 'w') as f:
# #             json.dump(output_data, f, indent=2)
        
# #         print(f"\n=== JSON Output Saved ===")
# #         print(f"File: {output_path}")
# #         print(f"Session duration: {session_end - session_start:.2f} seconds")
# #         print(f"Total gaze points: {len(self.gaze_data)}")
        
# #         return output_path

# # # Example usage
# # if __name__ == "__main__":
# #     # Load your events data from CSV
# #     events_df = pd.read_csv(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\csv\6.csv")
    
# #     # Fill empty demographic cells
# #     events_df['Gender'] = events_df['Gender'].fillna(method='ffill')
# #     events_df['Age'] = events_df['Age'].fillna(method='ffill') 
# #     events_df['File Start'] = events_df['File Start'].fillna(method='ffill')
    
# #     # Create simulator with CLUSTERING
# #     # CLUSTER_SIZE=50: Points within 50px of each other (like words)
# #     # FIXATION_DURATION=0.3: Average 300ms fixations (like reading)
# #     simulator = EEGGazeSimulatorClustered(
# #         edf_path=r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\edf\abnormal\0000006.edf",
# #         events_data=events_df,
# #         cluster_size=50,        # Like word size (50px clusters)
# #         fixation_duration=0.3   # 300ms average fixations
# #     )
    
# #     # Create JSON output
# #     output_file = simulator.create_json_output(
# #         output_path="D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/simulation_gazepoints/0000006_CLUSTERED.json",
# #         time_window=5.0
# #     )
    
# #     # Print sample gaze points to verify clustering
# #     print("\n=== Sample Clustered Gaze Points ===")
# #     for i, gaze in enumerate(simulator.gaze_data[:10]):
# #         print(f"\nPoint {i}:")
# #         print(f"  Time: {gaze['time']:.2f}s")
# #         print(f"  Channel: {gaze['channel']}")
# #         print(f"  Type: {gaze['gaze_type']}")
# #         print(f"  Raw coords: ({gaze['raw']['x']}, {gaze['raw']['y']})")
# #         print(f"  EEG coords: ({gaze['coords']['x']:.1f}, {gaze['coords']['y']:.0f})")
# #         print(f"  Cluster: {gaze['cluster_id']}")



import pyedflib
import numpy as np
import json
import pandas as pd
import random
from datetime import datetime, timedelta
import os

class EEGGazeSimulatorFixed:
    def __init__(self, edf_path, events_data):
        # ==== Load EDF ====
        self.f = pyedflib.EdfReader(edf_path)
        self.n_signals = self.f.signals_in_file
        self.labels = self.f.getSignalLabels()
        self.sample_rate = int(self.f.getSampleFrequency(0))
        self.signals = [self.f.readSignal(i) for i in range(self.n_signals)]
        self.duration = len(self.signals[0]) / self.sample_rate
        self.f.close()
        
        # Store edf filename
        self.edf_filename = os.path.basename(edf_path)

        print(f"=== EEG INFO ===")
        print(f"EEG Duration: {self.duration:.2f} seconds")
        print(f"EEG Sample Rate: {self.sample_rate} Hz")
        print(f"Number of channels: {self.n_signals}")
        print(f"EDF File: {self.edf_filename}")

        # ==== Process Events Data ====
        self.events_data = events_data
        self.events = self.process_events_data()
        
        print(f"\n=== EVENTS INFO ===")
        print(f"Loaded {len(self.events)} abnormal events")
        
        if self.events:
            total_event_duration = sum(e['duration'] for e in self.events)
            print(f"Total event duration: {total_event_duration:.1f}s ({total_event_duration/self.duration*100:.1f}% of EEG)")
            print("\nFirst 5 abnormal events:")
            for i, event in enumerate(self.events[:5]):
                print(f"  Event {i}: {event['start']:.2f}-{event['end']:.2f}s ({event['duration']:.2f}s) on {event['channels']}")

        # ==== FIXED: Better gaze parameters ====
        self.GAZE_SAMPLING_RATE = 33  # Hz - typical eye tracker
        
        # FIXED: Reduce normal density, increase event density
        self.NORMAL_GAZE_DENSITY = 0.2  # Was 0.3 - fewer normal points
        self.EVENT_GAZE_DENSITY = 1.2   # Was 1.0 - more event points
        
        # FIXED: Realistic coordinate ranges
        self.offset_step = 60  # Was 200 - channels closer together
        self.SCREEN_WIDTH = 1920
        self.SCREEN_HEIGHT = 1080
        
        # FIXED: Clustering parameters
        self.NORMAL_CLUSTER_RADIUS = 40  # Normal points within ±40px
        self.EVENT_CLUSTER_RADIUS = 20   # Event points within ±20px (tighter!)
        
        print(f"\n=== GAZE PARAMETERS ===")
        print(f"Normal cluster radius: ±{self.NORMAL_CLUSTER_RADIUS}px")
        print(f"Event cluster radius: ±{self.EVENT_CLUSTER_RADIUS}px")
        print(f"Your fixation dispersion threshold: 80px")
        print(f"-> Event clusters will DEFINITELY form fixations (20px < 80px)")
        print(f"-> Normal clusters SHOULD form fixations (40px < 80px)")
        
        # Generate gaze data
        self.gaze_data = self.generate_fixed_gaze_data()

    def process_events_data(self):
        """Process events data and map to EDF timeline"""
        events = []
        
        file_start_time_str = None
        for _, event in self.events_data.iterrows():
            if pd.notna(event['File Start']):
                file_start_time_str = event['File Start']
                break
        
        if file_start_time_str is None:
            file_start_seconds = 0
        else:
            file_start_seconds = self.time_string_to_seconds(file_start_time_str)
        
        for _, event in self.events_data.iterrows():
            if pd.isna(event['Start time']):
                continue
                
            start_sec = self.time_string_to_seconds(event['Start time'])
            end_sec = self.time_string_to_seconds(event['End time'])
            
            start_relative = start_sec - file_start_seconds
            end_relative = end_sec - file_start_seconds
            
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
        
        return valid_events
    
    def time_string_to_seconds(self, time_str):
        """Convert time string to total seconds"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 4:
                    hours, minutes, seconds, milliseconds = parts
                    return (int(hours) * 3600 + int(minutes) * 60 + 
                            int(seconds) + int(milliseconds) / 1000)
                elif len(parts) == 3:
                    hours, minutes, seconds = parts
                    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                else:
                    return float(time_str)
            else:
                return float(time_str)
        except:
            return 0.0

    def get_normal_periods(self, events):
        """Identify normal (non-event) periods"""
        normal_periods = []
        
        if not events:
            normal_periods.append({
                'start': 0,
                'end': self.duration,
                'duration': self.duration,
                'type': 'normal'
            })
        else:
            current_time = 0
            
            for event in sorted(events, key=lambda x: x['start']):
                if event['start'] > current_time:
                    normal_periods.append({
                        'start': current_time,
                        'end': event['start'],
                        'duration': event['start'] - current_time,
                        'type': 'normal'
                    })
                current_time = max(current_time, event['end'])
            
            if current_time < self.duration:
                normal_periods.append({
                    'start': current_time,
                    'end': self.duration,
                    'duration': self.duration - current_time,
                    'type': 'normal'
                })
        
        # Keep periods longer than 0.5s
        normal_periods = [p for p in normal_periods if p['duration'] >= 0.5]
        
        return normal_periods

    def generate_normal_gaze_points(self, normal_periods):
        """Generate gaze points for normal periods - WITH CLUSTERING"""
        all_gaze_points = []
        
        total_normal_duration = sum(p['duration'] for p in normal_periods)
        total_points = int(total_normal_duration * self.GAZE_SAMPLING_RATE * self.NORMAL_GAZE_DENSITY)
        
        print(f"\n=== NORMAL GAZE ===")
        print(f"Total normal duration: {total_normal_duration:.1f}s")
        print(f"Total normal points: {total_points}")
        
        # We'll create "fixation clusters" for normal periods
        # Each cluster = one potential fixation
        num_clusters = max(10, int(total_normal_duration * 0.5))  # ~0.5 clusters per second
        
        for cluster_id in range(num_clusters):
            # Choose a random normal period for this cluster
            period = random.choice(normal_periods)
            
            # Random time within this period
            cluster_time = np.random.uniform(period['start'], period['end'])
            
            # Choose a channel for this cluster
            channel_idx = np.random.randint(0, len(self.labels))
            channel_name = self.labels[channel_idx]
            
            # Base position for this channel
            base_y = channel_idx * self.offset_step
            
            # Create 5-15 points in this cluster (like a fixation)
            points_in_cluster = random.randint(5, 15)
            
            for point_num in range(points_in_cluster):
                # Time slightly varies within cluster
                point_time = cluster_time + (point_num * 0.03)  # 30ms spacing
                
                # FIXED: TIGHT CLUSTERING around base position
                x_jitter = np.random.normal(0, 0.02)  # ±20ms in time
                y_jitter = np.random.normal(0, self.NORMAL_CLUSTER_RADIUS/2)  # Tight!
                
                x_coord = point_time + x_jitter
                y_coord = base_y + y_jitter
                
                # Get EEG value
                eeg_value = self.get_eeg_value(channel_name, point_time)
                
                # Create gaze point
                gaze_point = self.create_gaze_point(
                    timestamp=point_time,
                    eeg_time=point_time,
                    channel_name=channel_name,
                    eeg_value=eeg_value,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    gaze_type='normal',
                    cluster_id=cluster_id
                )
                all_gaze_points.append(gaze_point)
        
        return all_gaze_points

    def generate_abnormal_gaze_points(self, events):
        """Generate gaze points for abnormal events - WITH TIGHT CLUSTERING"""
        all_gaze_points = []
        
        print(f"\n=== ABNORMAL GAZE ===")
        
        for event_idx, event in enumerate(events):
            event_channels = [ch for ch in event['channels'] if ch in self.labels]
            
            if not event_channels:
                continue
            
            # Calculate points for this event
            event_points = int(event['duration'] * self.GAZE_SAMPLING_RATE * self.EVENT_GAZE_DENSITY)
            event_points = max(20, min(event_points, 2000))
            
            print(f"  Event {event_idx}: {event['duration']:.1f}s -> {event_points} points")
            
            # Create MULTIPLE tight clusters for this event
            # Each cluster = one fixation
            num_clusters = max(3, int(event['duration'] * 3))  # 3 clusters per second
            
            for cluster_num in range(num_clusters):
                # Time within event
                event_center = (event['start'] + event['end']) / 2
                time_std = event['duration'] / 6  # Tighter time clustering
                cluster_time = np.random.normal(event_center, time_std)
                cluster_time = max(event['start'], min(event['end'], cluster_time))
                
                # Choose an event channel (90% on event channels)
                if np.random.random() < 0.9:
                    channel_name = random.choice(event_channels)
                else:
                    channel_name = random.choice(self.labels)
                
                channel_idx = self.labels.index(channel_name)
                base_y = channel_idx * self.offset_step
                
                # Create 8-20 points in this VERY TIGHT cluster
                points_in_cluster = random.randint(8, 20)
                
                for point_num in range(points_in_cluster):
                    point_time = cluster_time + (point_num * 0.03)
                    
                    # FIXED: VERY TIGHT CLUSTERING
                    x_jitter = np.random.normal(0, 0.01)  # ±10ms
                    y_jitter = np.random.normal(0, self.EVENT_CLUSTER_RADIUS/3)  # Super tight!
                    
                    x_coord = point_time + x_jitter
                    y_coord = base_y + y_jitter
                    
                    # Get EEG value
                    eeg_value = self.get_eeg_value(channel_name, point_time)
                    
                    # Create gaze point
                    gaze_point = self.create_gaze_point(
                        timestamp=point_time,
                        eeg_time=point_time,
                        channel_name=channel_name,
                        eeg_value=eeg_value,
                        x_coord=x_coord,
                        y_coord=y_coord,
                        gaze_type='abnormal',
                        cluster_id=f"e{event_idx}_c{cluster_num}"
                    )
                    all_gaze_points.append(gaze_point)
        
        return all_gaze_points

    def get_eeg_value(self, channel_name, time):
        """Get EEG value at specific time"""
        try:
            channel_idx = self.labels.index(channel_name)
            sample_idx = int(time * self.sample_rate)
            
            if 0 <= sample_idx < len(self.signals[channel_idx]):
                return float(self.signals[channel_idx][sample_idx])
        except:
            pass
        
        return None

    def create_gaze_point(self, timestamp, eeg_time, channel_name, eeg_value, 
                         x_coord, y_coord, gaze_type, cluster_id):
        """Create a gaze point with REALISTIC coordinates"""
        base_timestamp = datetime.now().timestamp() - self.duration
        absolute_timestamp = base_timestamp + timestamp
        
        # FIXED: Realistic screen coordinates
        # Map y_coord (0-~2000) to screen height (1080)
        screen_y = int(self.SCREEN_HEIGHT * 0.2 + 
                      (y_coord / (len(self.labels) * self.offset_step)) * self.SCREEN_HEIGHT * 0.6)
        
        # Map x_coord (time) to screen width
        screen_x = int(self.SCREEN_WIDTH * 0.2 + 
                      (x_coord / self.duration) * self.SCREEN_WIDTH * 0.6)
        
        # Keep within screen bounds
        screen_x = max(100, min(screen_x, self.SCREEN_WIDTH - 100))
        screen_y = max(100, min(screen_y, self.SCREEN_HEIGHT - 100))
        
        return {
            "timestamp": absolute_timestamp,
            "time": float(eeg_time),
            "channel": channel_name,
            "value": eeg_value,
            "coords": {
                "x": float(x_coord),
                "y": float(y_coord)
            },
            "raw": {
                "x": screen_x,
                "y": screen_y
            },
            "gaze_type": gaze_type,
            "cluster_id": str(cluster_id),
            "metadata": {
                "sampling_rate": self.GAZE_SAMPLING_RATE,
                "eeg_sample_rate": self.sample_rate
            }
        }

    def generate_fixed_gaze_data(self):
        """Generate gaze data with proper clustering"""
        print(f"\n=== Generating FIXED Gaze Data ===")
        
        normal_periods = self.get_normal_periods(self.events)
        
        total_event_duration = sum(e['duration'] for e in self.events)
        total_normal_duration = sum(p['duration'] for p in normal_periods)
        
        print(f"Total EEG: {self.duration:.1f}s")
        print(f"Event duration: {total_event_duration:.1f}s")
        print(f"Normal duration: {total_normal_duration:.1f}s")
        
        # Generate points
        normal_gaze = self.generate_normal_gaze_points(normal_periods)
        abnormal_gaze = self.generate_abnormal_gaze_points(self.events)
        
        all_gaze = normal_gaze + abnormal_gaze
        all_gaze.sort(key=lambda x: x['timestamp'])
        
        # Calculate statistics
        normal_clusters = len(set(g['cluster_id'] for g in normal_gaze))
        abnormal_clusters = len(set(g['cluster_id'] for g in abnormal_gaze))
        
        print(f"\n=== Generation Complete ===")
        print(f"Normal points: {len(normal_gaze)} in {normal_clusters} clusters")
        print(f"Abnormal points: {len(abnormal_gaze)} in {abnormal_clusters} clusters")
        print(f"Total points: {len(all_gaze)}")
        
        # Check coordinate ranges
        xs = [p['coords']['x'] for p in all_gaze]
        ys = [p['coords']['y'] for p in all_gaze]
        raw_xs = [p['raw']['x'] for p in all_gaze]
        raw_ys = [p['raw']['y'] for p in all_gaze]
        
        print(f"\nCoordinate Ranges:")
        print(f"  EEG X (time): {min(xs):.1f} - {max(xs):.1f}s")
        print(f"  EEG Y: {min(ys):.0f} - {max(ys):.0f}px")
        print(f"  Raw X: {min(raw_xs)} - {max(raw_xs)}px")
        print(f"  Raw Y: {min(raw_ys)} - {max(raw_ys)}px")
        
        # Predict fixations
        print(f"\n=== PREDICTED FIXATIONS ===")
        print(f"Your dispersion threshold: 80px")
        print(f"Normal cluster radius: ±{self.NORMAL_CLUSTER_RADIUS}px -> WILL form fixations")
        print(f"Event cluster radius: ±{self.EVENT_CLUSTER_RADIUS}px -> WILL form fixations")
        print(f"Expected fixations: ~{normal_clusters + abnormal_clusters}")
        
        return all_gaze

    def create_json_output(self, output_path=None, time_window=10.0):
        """Create JSON output"""
        if output_path is None:
            base_name = os.path.splitext(self.edf_filename)[0]
            output_path = f"simulated_gaze_fixed_{base_name}.json"
        
        session_start = min(gaze['timestamp'] for gaze in self.gaze_data)
        session_end = max(gaze['timestamp'] for gaze in self.gaze_data)
        
        # Count clusters
        normal_points = [g for g in self.gaze_data if g['gaze_type'] == 'normal']
        abnormal_points = [g for g in self.gaze_data if g['gaze_type'] == 'abnormal']
        normal_clusters = len(set(g['cluster_id'] for g in normal_points))
        abnormal_clusters = len(set(g['cluster_id'] for g in abnormal_points))
        
        output_data = {
            "edf_file": self.edf_filename,
            "simulation_info": {
                "normal_clusters": normal_clusters,
                "abnormal_clusters": abnormal_clusters,
                "total_clusters": normal_clusters + abnormal_clusters,
                "normal_cluster_radius_px": self.NORMAL_CLUSTER_RADIUS,
                "event_cluster_radius_px": self.EVENT_CLUSTER_RADIUS,
                "your_dispersion_threshold": 80
            },
            "session": [
                {
                    "time_window": float(time_window),
                    "time_window_start": 0.0,
                    "start_time": float(session_start),
                    "end_time": float(session_end),
                    "gaze_data": self.gaze_data
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n=== JSON Output Saved ===")
        print(f"File: {output_path}")
        print(f"Expected clusters/fixations: {normal_clusters} normal, {abnormal_clusters} abnormal")
        
        return output_path

# Example usage
if __name__ == "__main__":
    # Load events data
    events_df = pd.read_csv(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\csv\6.csv")
    
    # Fill empty cells
    events_df['Gender'] = events_df['Gender'].fillna(method='ffill')
    events_df['Age'] = events_df['Age'].fillna(method='ffill') 
    events_df['File Start'] = events_df['File Start'].fillna(method='ffill')
    
    # FIXED simulator
    simulator = EEGGazeSimulatorFixed(
        edf_path=r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\edf\abnormal\0000006.edf",
        events_data=events_df
    )
    
    # Create JSON output
    output_file = simulator.create_json_output(
        output_path="D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/simulation_gazepoints/0000006_FIXED.json",
        time_window=5.0
    )