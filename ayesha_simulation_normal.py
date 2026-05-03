import pyedflib
import numpy as np
import json
import random
from datetime import datetime
import os

class SingleFileNormalGazeWithFixations:
    def __init__(self, edf_path, output_path=None):
        """
        Process a SINGLE normal EEG file with fixation generation
        """
        self.edf_path = edf_path
        self.edf_filename = os.path.basename(edf_path)
        
        # Set output path
        if output_path is None:
            base_name = os.path.splitext(self.edf_filename)[0]
            self.output_path = f"gaze_{base_name}.json"
        else:
            self.output_path = output_path
        
        print(f"\n{'='*60}")
        print(f"PROCESSING SINGLE FILE: {self.edf_filename}")
        print(f"{'='*60}")
        
        # Load EDF file
        self.load_edf()
        
        # Generate gaze with fixations
        self.gaze_data = self.generate_gaze_with_fixations()
        
        # Save to JSON
        self.save_to_json()
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'='*60}")
    
    def load_edf(self):
        """Load EEG data from EDF file"""
        print(f"\n[1/3] Loading EEG data...")
        
        f = pyedflib.EdfReader(self.edf_path)
        self.labels = f.getSignalLabels()
        self.sample_rate = int(f.getSampleFrequency(0))
        self.signals = [f.readSignal(i) for i in range(len(self.labels))]
        self.duration = len(self.signals[0]) / self.sample_rate
        f.close()
        
        print(f"   ✓ Duration: {self.duration:.1f} seconds")
        print(f"   ✓ Channels: {len(self.labels)}")
        print(f"   ✓ Sample rate: {self.sample_rate} Hz")
        print(f"   ✓ File size: {self.duration/60:.1f} minutes")
    
    def generate_gaze_with_fixations(self):
        """Generate gaze points with REAL fixations"""
        print(f"\n[2/3] Generating gaze with fixations...")
        
        # ============================================
        # FIXATION PARAMETERS (WILL CREATE FIXATIONS!)
        # ============================================
        FIXATION_DURATION_RANGE = (0.2, 1.5)     # Fixations last 0.2-1.5 seconds
        POINTS_PER_FIXATION = 8                  # Points in each fixation
        CLUSTER_RADIUS = 35                      # pixels (WILL create fixations!)
        
        # 9 fixation areas on screen (1920x1080)
        FIXATION_CENTERS = [
            (400, 300),    # Top-left
            (1000, 350),   # Top-center
            (1600, 320),   # Top-right
            (450, 600),    # Middle-left
            (1050, 650),   # Middle-center
            (1650, 620),   # Middle-right
            (500, 900),    # Bottom-left
            (1100, 850),   # Bottom-center
            (1550, 880),   # Bottom-right
        ]
        
        # ============================================
        # GENERATE GAZE POINTS
        # ============================================
        gaze_points = []
        current_time = 0
        fixation_count = 0
        saccade_count = 0
        
        print(f"   Generating natural eye movement pattern...")
        print(f"   • 70% fixations (clustered points)")
        print(f"   • 30% saccades (quick movements)")
        print(f"   • Cluster radius: {CLUSTER_RADIUS}px")
        print(f"   • Your detection threshold: 80px")
        print(f"   → {CLUSTER_RADIUS}px < 80px = WILL CREATE FIXATIONS!")
        
        # Continue until we cover the EEG duration
        while current_time < self.duration:
            # Decide if this is a fixation or saccade
            is_fixation = random.random() > 0.3  # 70% fixations, 30% saccades
            
            if is_fixation:
                # ============================================
                # CREATE A FIXATION CLUSTER
                # ============================================
                fixation_duration = random.uniform(*FIXATION_DURATION_RANGE)
                fixation_duration = min(fixation_duration, self.duration - current_time)
                
                if fixation_duration < 0.2:  # Too short
                    current_time += random.uniform(0.05, 0.15)
                    continue
                
                # Choose a fixation center
                center_x, center_y = random.choice(FIXATION_CENTERS)
                
                # Choose a channel (prefer central channels)
                central_channels = ['C3', 'C4', 'Cz', 'Fz', 'Pz', 'O1', 'O2']
                available_channels = [ch for ch in self.labels if ch in central_channels]
                if available_channels:
                    channel = random.choice(available_channels)
                else:
                    channel = random.choice(self.labels)
                
                # Create CLUSTERED points for this fixation
                for i in range(POINTS_PER_FIXATION):
                    point_time = current_time + (i * fixation_duration / POINTS_PER_FIXATION)
                    
                    if point_time >= self.duration:
                        break
                    
                    # GAUSSIAN CLUSTERING around center (tight grouping!)
                    screen_x = int(np.random.normal(center_x, CLUSTER_RADIUS/3))
                    screen_y = int(np.random.normal(center_y, CLUSTER_RADIUS/3))
                    
                    # Keep within screen bounds (1920x1080)
                    screen_x = max(50, min(screen_x, 1870))
                    screen_y = max(50, min(screen_y, 1030))
                    
                    # Get EEG value
                    eeg_value = self.get_eeg_value(channel, point_time)
                    
                    # Create timestamp
                    base_ts = datetime.now().timestamp() - self.duration
                    timestamp = base_ts + point_time
                    
                    # EEG coordinates for visualization
                    channel_idx = self.labels.index(channel)
                    eeg_y = channel_idx * 60
                    
                    gaze_points.append({
                        "timestamp": timestamp,
                        "time": float(point_time),
                        "channel": channel,
                        "value": eeg_value,
                        "coords": {
                            "x": float(point_time),
                            "y": float(eeg_y)
                        },
                        "raw": {
                            "x": screen_x,
                            "y": screen_y
                        },
                        "fixation_id": f"fix_{fixation_count}",
                        "metadata": {
                            "channel_index": channel_idx,
                            "is_fixation": True,
                            "cluster_radius": CLUSTER_RADIUS,
                            "center_x": center_x,
                            "center_y": center_y
                        }
                    })
                
                fixation_count += 1
                current_time += fixation_duration
                
            else:
                # ============================================
                # CREATE A SACCADE (quick movement)
                # ============================================
                saccade_duration = random.uniform(0.05, 0.15)
                
                # Choose random start and end centers
                start_center = random.choice(FIXATION_CENTERS)
                end_center = random.choice([c for c in FIXATION_CENTERS if c != start_center])
                
                # Create 2-3 points along saccade path
                saccade_points = random.randint(2, 3)
                
                for i in range(saccade_points):
                    point_time = current_time + (i * saccade_duration / saccade_points)
                    
                    if point_time >= self.duration:
                        break
                    
                    # Interpolate between start and end
                    progress = i / saccade_points
                    screen_x = int(start_center[0] + (end_center[0] - start_center[0]) * progress)
                    screen_y = int(start_center[1] + (end_center[1] - start_center[1]) * progress)
                    
                    # Add movement jitter
                    screen_x += random.randint(-50, 50)
                    screen_y += random.randint(-50, 50)
                    
                    # Keep within screen bounds
                    screen_x = max(50, min(screen_x, 1870))
                    screen_y = max(50, min(screen_y, 1030))
                    
                    # Random channel during saccade
                    channel = random.choice(self.labels)
                    
                    # Get EEG value
                    eeg_value = self.get_eeg_value(channel, point_time)
                    
                    # Create timestamp
                    base_ts = datetime.now().timestamp() - self.duration
                    timestamp = base_ts + point_time
                    
                    # EEG coordinates
                    channel_idx = self.labels.index(channel)
                    eeg_y = channel_idx * 60
                    
                    gaze_points.append({
                        "timestamp": timestamp,
                        "time": float(point_time),
                        "channel": channel,
                        "value": eeg_value,
                        "coords": {
                            "x": float(point_time),
                            "y": float(eeg_y)
                        },
                        "raw": {
                            "x": screen_x,
                            "y": screen_y
                        },
                        "fixation_id": None,
                        "metadata": {
                            "channel_index": channel_idx,
                            "is_fixation": False,
                            "movement_type": "saccade"
                        }
                    })
                
                saccade_count += 1
                current_time += saccade_duration
        
        # Sort by time
        gaze_points.sort(key=lambda x: x['time'])
        
        # Count points
        fixation_points = [g for g in gaze_points if g['metadata']['is_fixation']]
        saccade_points = [g for g in gaze_points if not g['metadata']['is_fixation']]
        
        print(f"\n   ✓ Generated {len(gaze_points)} total points")
        print(f"   ✓ Created {fixation_count} fixation clusters")
        print(f"   ✓ Created {saccade_count} saccades")
        print(f"   ✓ Fixation points: {len(fixation_points)} (clustered)")
        print(f"   ✓ Saccade points: {len(saccade_points)} (scattered)")
        print(f"\n   FIXATION GUARANTEE:")
        print(f"   • Cluster radius: {CLUSTER_RADIUS}px")
        print(f"   • Dispersion: ~{CLUSTER_RADIUS*2}px")
        print(f"   • Your threshold: 80px")
        print(f"   • {CLUSTER_RADIUS*2}px < 80px = FIXATIONS WILL BE DETECTED!")
        
        return gaze_points
    
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
    
    def save_to_json(self):
        """Save gaze data to JSON file"""
        print(f"\n[3/3] Saving to JSON...")
        
        session_start = min(g['timestamp'] for g in self.gaze_data)
        session_end = max(g['timestamp'] for g in self.gaze_data)
        
        # Count statistics
        fixation_points = [g for g in self.gaze_data if g['metadata']['is_fixation']]
        saccade_points = [g for g in self.gaze_data if not g['metadata']['is_fixation']]
        fixation_ids = set(g['fixation_id'] for g in self.gaze_data if g['fixation_id'])
        
        output_data = {
            "edf_file": self.edf_filename,
            "simulation_info": {
                "total_points": len(self.gaze_data),
                "fixation_clusters": len(fixation_ids),
                "fixation_points": len(fixation_points),
                "saccade_points": len(saccade_points),
                "cluster_radius_px": 35,
                "fixation_dispersion": f"~70px (35px radius * 2)",
                "detection_threshold": 80,
                "fixation_guarantee": "YES (70px < 80px)",
                "duration_seconds": float(self.duration),
                "channels_used": len(set(g['channel'] for g in self.gaze_data)),
                "generation_method": "clustered_fixations",
                "generation_timestamp": datetime.now().isoformat()
            },
            "session": [{
                "time_window": 5.0,
                "time_window_start": 0.0,
                "start_time": float(session_start),
                "end_time": float(session_end),
                "gaze_data": self.gaze_data
            }]
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"   ✓ Saved to: {self.output_path}")
        print(f"   ✓ File size: {os.path.getsize(self.output_path)/1024:.1f} KB")
        print(f"\n   JSON CONTAINS:")
        print(f"   • {len(fixation_ids)} fixations (clustered points)")
        print(f"   • Each fixation: 8 points in {35}px radius")
        print(f"   • Your fixation detector WILL find them!")


# ============================================
# TEST FUNCTION
# ============================================

def test_single_file():
    """Test the single file processor"""
    
    print("="*60)
    print("SINGLE FILE TEST - NORMAL EEG WITH FIXATIONS")
    print("="*60)
    
    # ============================================
    # CONFIGURATION - CHANGE THESE!
    # ============================================
    
    # Example EDF file (change to your file)
    TEST_EDF = r"E:\NMT_events\NMT_events\edf\Normal EDF Files\0000001.edf"
    
    # Or use a test file from your abnormal folder
    # TEST_EDF = r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\edf\abnormal\0000006.edf"
    
    # Output path (optional - will auto-generate if None)
    OUTPUT_JSON = None  # Will create "gaze_0000001.json"
    # OUTPUT_JSON = "test_output.json"  # Or specify custom name
    
    # ============================================
    # RUN THE TEST
    # ============================================
    
    # Check if file exists
    if not os.path.exists(TEST_EDF):
        print(f"\nERROR: File not found!")
        print(f"Please check: {TEST_EDF}")
        
        # Try to find any EDF file
        import glob
        test_files = glob.glob(r"E:\NMT_events\NMT_events\edf\Normal EDF Files\*.edf")
        if test_files:
            print(f"\nFound these EDF files:")
            for f in test_files[:5]:  # Show first 5
                print(f"  • {os.path.basename(f)}")
            print(f"\nPlease update TEST_EDF path in the code.")
        return
    
    print(f"\nInput EDF: {TEST_EDF}")
    print(f"Output JSON: {OUTPUT_JSON or 'Auto-generated'}")
    print(f"\n{'='*60}")
    
    # Create and run the processor
    processor = SingleFileNormalGazeWithFixations(
        edf_path=TEST_EDF,
        output_path=OUTPUT_JSON
    )
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE!")
    print(f"{'='*60}")
    print(f"\nNEXT STEPS:")
    print(f"1. Run your fixation detection on: {processor.output_path}")
    print(f"2. Expected: ~{len(processor.gaze_data)//8} fixations detected")
    print(f"3. Each fixation has points within {35}px radius")
    print(f"4. Your 80px threshold WILL detect them!")


# ============================================
# QUICK TEST WITH MINIMAL CODE
# ============================================

def quick_test():
    """Even simpler test - just paste your file path"""
    
    print("\n" + "="*60)
    print("QUICK TEST - Paste your EDF file path below")
    print("="*60)
    
    # PASTE YOUR EDF FILE PATH HERE:
    YOUR_EDF_FILE = r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\edf\normal\0000001.edf"
    
    if not os.path.exists(YOUR_EDF_FILE):
        print(f"\nFile not found: {YOUR_EDF_FILE}")
        print("Please update the path in the code.")
        return
    
    # Process it
    print(f"\nProcessing: {os.path.basename(YOUR_EDF_FILE)}")
    
    processor = SingleFileNormalGazeWithFixations(
        edf_path=YOUR_EDF_FILE,
        output_path="test_gaze_output.json"
    )
    
    print(f"\n✓ Output saved to: {processor.output_path}")
    print("✓ This file WILL contain detectable fixations!")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # ============================================
    # CHOOSE WHICH TEST TO RUN:
    # ============================================
    
    # Option 1: Full test with details
    test_single_file()
    
