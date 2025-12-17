import numpy as np
import pyedflib
import json
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QListWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

class ChannelSelectorDialog(QDialog):
    def __init__(self, channels, selected_channels):
        super().__init__()
        self.setWindowTitle("Select Channels to Display")
        self.selected = selected_channels.copy()
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        for ch in channels:
            self.list_widget.addItem(ch)
        for i in range(self.list_widget.count()):
            if channels[i] in selected_channels:
                self.list_widget.item(i).setSelected(True)
        layout.addWidget(self.list_widget)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.accept)
        layout.addWidget(apply_btn)
        self.setLayout(layout)

    def get_selected(self):
        return [item.text() for item in self.list_widget.selectedItems()]

import numpy as np
import pyedflib
import json
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QListWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

class EDFPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.fig.set_size_inches(10, 4)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.signals = []
        self.labels = []
        self.sample_rate = 0
        self.current_time = 0
        self.time_window = 10
        self.selected_channels = []
        self.amplitude_scale = 245
        
        # Heatmap attributes
        self.heatmap_loaded = False
        self.heatmap_file_path = None
        self.heatmap_synthetic = False
        self.fixations = []
        self.heatmap_sigma = 10
        self.heatmap_alpha = 0.6
        self.show_heatmap = False

    def load_edf(self, file_path):
        self.ax.clear()
        try:
            f = pyedflib.EdfReader(file_path)
            self.labels = f.getSignalLabels()
            self.sample_rate = f.getSampleFrequency(0)
            self.duration = f.getFileDuration()
            self.signals = [f.readSignal(i) for i in range(f.signals_in_file)]
            self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
            f.close()
            self.selected_channels = self.labels.copy()
            self.current_time = 0
            self.plot_current_window()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not open EDF file:\n{e}")

    def load_heatmap_data(self, file_path=None, synthetic=False):
        """Load fixation data for heatmap generation"""
        self.heatmap_loaded = True
        self.heatmap_file_path = file_path
        self.heatmap_synthetic = synthetic
        self.fixations = []
        
        try:
            if file_path and not synthetic:
                print(f"Loading heatmap data from: {file_path}")
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Extract fixations from session data
                if "sessions" in data:
                    for session in data["sessions"]:
                        if "fixations" in session:
                            for fx in session["fixations"]:
                                fx["session_id"] = session.get("session_id", "unknown")
                                self.fixations.append(fx)
                elif "fixations" in data:
                    self.fixations = data["fixations"]
                elif "session" in data:  # Your current session format
                    print("Found session data format")
                    for session in data["session"]:
                        if "gaze_data" in session:
                            # Convert gaze data to fixations
                            for gaze in session["gaze_data"]:
                                if gaze.get("coords") and gaze["coords"]["x"] is not None:
                                    fixation = {
                                        "x": gaze["coords"]["x"],
                                        "y": gaze["coords"]["y"],
                                        "duration": 0.1,  # default duration
                                        "channel": [gaze["channel"]] if gaze["channel"] else [],
                                        "start_time": gaze.get("timestamp", 0),
                                        "end_time": gaze.get("timestamp", 0) + 0.1
                                    }
                                    self.fixations.append(fixation)
            
            elif synthetic:
                # Generate synthetic fixations for testing
                self.generate_synthetic_fixations()
                
            print(f"Loaded {len(self.fixations)} fixations for heatmap")
            print(f"First few fixations: {self.fixations[:3]}")
            self.show_heatmap = True
            
        except Exception as e:
            print(f"Error loading heatmap data: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not load heatmap data:\n{e}")

    def get_fixations_in_current_window(self):
        """Return fixations within the current time window"""
        if not self.fixations:
            return []
            
        window_start = self.current_time
        window_end = self.current_time + self.time_window
        current_fixations = []

        for fx in self.fixations:
            if fx.get("x") is None:
                continue

            fixation_time = fx["x"]
            fixation_duration = fx.get("duration", 0.1)

            if window_start <= fixation_time <= window_end:
                current_fixations.append({
                    "timestamp": fixation_time,
                    "duration": fixation_duration,
                    "channel": fx.get("channel", []),
                    "y": fx.get("y", 0),
                    "session": fx.get("session_id", "")
                })

        print(f"Found {len(current_fixations)} fixations in current window")
        return current_fixations

    def create_heatmap_for_window(self, current_fixations, channel_offsets):
        """Create a heatmap for the current window's fixations"""
        if not current_fixations:
            return None

        grid_w, grid_h = 200, 100
        heatmap = np.zeros((grid_h, grid_w), dtype=float)
        
        # Calculate window dimensions
        window_height = self.offset_step * max(1, len(self.selected_channels))

        for fx in current_fixations:
            # Match fixation channel to EEG label offset
            offset = None
            for label, off in channel_offsets.items():
                if any(label == ch for ch in fx["channel"]):
                    offset = off
                    break

            if offset is None:
                offset = window_height / 2

            time_normalized = (fx["timestamp"] - self.current_time) / self.time_window
            channel_normalized = offset / window_height
            x_idx = int(time_normalized * (grid_w - 1))
            y_idx = int(channel_normalized * (grid_h - 1))

            if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
                heatmap[y_idx, x_idx] += fx["duration"]

        if np.max(heatmap) > 0:
            heatmap_smooth = gaussian_filter(heatmap, sigma=self.heatmap_sigma)
            heatmap_norm = (heatmap_smooth - np.min(heatmap_smooth)) / (np.max(heatmap_smooth) - np.min(heatmap_smooth))
            return heatmap_norm
        return None

    def draw_heatmap(self):
        """Draw heatmap overlay on the current plot"""
        if not self.show_heatmap or not self.heatmap_loaded:
            return
            
        # Get channel offsets for current plot
        channel_offsets = {}
        for i, label in enumerate(self.selected_channels):
            channel_offsets[label] = i * self.offset_step
            
        # Get fixations in current window
        current_fixations = self.get_fixations_in_current_window()
        
        if not current_fixations:
            print(f"No fixations in current window: {self.current_time} to {self.current_time + self.time_window}")
            return
            
        print(f"Drawing heatmap with {len(current_fixations)} fixations")
        
        # Create heatmap
        heatmap = self.create_heatmap_for_window(current_fixations, channel_offsets)
        
        if heatmap is not None:
            window_height = self.offset_step * max(1, len(self.selected_channels))
            extent = [self.current_time, self.current_time + self.time_window, 0, window_height]
            
            # Draw heatmap with proper zorder to be behind EEG traces
            self.ax.imshow(heatmap, extent=extent, origin='lower', cmap='hot',
                          alpha=self.heatmap_alpha, aspect='auto', zorder=1)
            print("Heatmap drawn successfully")

    def toggle_heatmap(self):
        """Toggle heatmap display on/off"""
        self.show_heatmap = not self.show_heatmap
        print(f"Heatmap display toggled: {self.show_heatmap}")
        self.plot_current_window()

    def set_heatmap_parameters(self, sigma=None, alpha=None):
        """Set heatmap visualization parameters"""
        if sigma is not None:
            self.heatmap_sigma = sigma
        if alpha is not None:
            self.heatmap_alpha = alpha
        if self.show_heatmap:
            self.plot_current_window()

    def plot_current_window(self, debug_colors=False):
        if not self.signals:
            return

        self.ax.clear()
        start_idx = int(self.current_time * self.sample_rate)
        end_idx = int((self.current_time + self.time_window) * self.sample_rate)

        offset_step = 200
        visible_channels = [label for label in self.labels if label in self.selected_channels]
        num_visible = len(visible_channels)

        self.offset_step = offset_step
        self.num_visible = len(visible_channels)

        # === DRAW HEATMAP FIRST (so it's behind EEG traces) ===
        if self.show_heatmap and self.heatmap_loaded:
            self.draw_heatmap()

        # === PLOT EEG TRACES ON TOP ===
        for i, label in enumerate(self.labels):
            if label not in self.selected_channels:
                continue
            sig = self.signals[i]
            segment = sig[start_idx:end_idx]
            times = np.linspace(self.current_time, self.current_time + self.time_window, len(segment))
            offset = visible_channels.index(label) * offset_step

            scaled_segment = segment * self.amplitude_scale

            if debug_colors:
                rgb = np.array(self.encode_color(i, debug=True)) / 255.0
                self.ax.plot(times, scaled_segment + offset, color=rgb, linewidth=1.5, zorder=3)
                label_x = self.current_time + self.time_window * 0.01
                self.ax.text(
                    label_x, offset, label,
                    va='center', ha='left', fontsize=9, fontweight='bold',
                    color=rgb,
                    bbox=dict(facecolor='white', alpha=0.9),
                    zorder=4
                )
            else:
                self.ax.plot(times, scaled_segment + offset, color='black', linewidth=1.0, zorder=3)
                label_x = self.current_time + self.time_window * 0.01
                self.ax.text(
                    label_x, offset, label,
                    va='center', ha='left', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9),
                    zorder=4
                )

        # Draw scale bar
        bar_x = self.current_time + self.time_window - 1.5
        bar_y = -100
        self.ax.plot([bar_x, bar_x], [bar_y, bar_y + 50], color='black', lw=2, zorder=4)
        self.ax.plot([bar_x, bar_x + 1], [bar_y, bar_y], color='black', lw=2, zorder=4)
        self.ax.text(bar_x + 0.1, bar_y + 25, "50 µV", va='center', zorder=4)
        self.ax.text(bar_x + 0.5, bar_y - 10, "1 s", ha='center', zorder=4)

        # Add time ticks
        num_ticks = 10
        time_ticks = np.linspace(self.current_time, self.current_time + self.time_window, num_ticks)
        time_labels = [f"{tick:.2f}" for tick in time_ticks]
        self.ax.set_xticks(time_ticks)
        self.ax.set_xticklabels(time_labels)

        # Set axes limits
        self.ax.set_xlim(self.current_time, self.current_time + self.time_window)
        self.ax.set_ylim(-150, offset_step * max(1, num_visible))

        # Hide y-axis, show x-axis
        self.ax.axis('on')
        self.ax.get_yaxis().set_visible(False)

        # Update title based on heatmap status
        if self.show_heatmap and self.heatmap_loaded:
            current_fixations = self.get_fixations_in_current_window()
            self.ax.set_title(f"EEG + Heatmap ({len(current_fixations)} fixations in view)", fontsize=12)
        else:
            self.ax.set_title("Clinical-style EEG Viewer", fontsize=12)

        self.draw()
        
        # Build hidden mask buffer for gaze lookup
        self.mask_buffer = self.make_channel_mask(start_idx, end_idx, visible_channels, offset_step)



    def encode_color(self, idx, debug=False):
        """Encode channel index into RGB color.
        If debug=True, return bright visible colors for plotting.
        Otherwise, return bit-packed encoding for mask lookup.
        """
        if debug:
            # Cycle through distinct visible colors
            base_colors = [
                (255, 0, 0),     # Red
                (0, 255, 0),     # Green
                (0, 0, 255),     # Blue
                (255, 255, 0),   # Yellow
                (255, 0, 255),   # Magenta
                (0, 255, 255),   # Cyan
                (255, 128, 0),   # Orange
                (128, 0, 255),   # Purple
            ]
            return base_colors[idx % len(base_colors)]
        else:
            # Original packed encoding
            r = (idx & 0xFF)
            g = (idx >> 8) & 0xFF
            b = (idx >> 16) & 0xFF
            return (r, g, b)


    def decode_channel_from_color(self, color):
        """Decode (R,G,B) back into channel index."""
        r, g, b = color
        idx = r + (g << 8) + (b << 16)
        if 0 <= idx < len(self.selected_channels):
            return idx
        return None

    def make_channel_mask(self, start_idx, end_idx, visible_channels, offset_step):
        """Render hidden mask image where each channel has a unique color."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as Agg

        fig = Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)

        for i, label in enumerate(visible_channels):
            sig = self.signals[self.labels.index(label)][start_idx:end_idx]
            times = np.linspace(self.current_time, self.current_time + self.time_window, len(sig))
            offset = i * offset_step
            color = np.array(self.encode_color(i, debug=False)) / 255.0
            ax.plot(times, sig + offset, color=color, linewidth=2)

        ax.set_xlim(self.current_time, self.current_time + self.time_window)
        ax.set_ylim(-150, offset_step * max(1, len(visible_channels)))
        ax.axis("off")

        # Render to numpy buffer
        canvas = Agg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        return buf[:, :, :3]  # drop alpha channel


    

    def scroll_time(self, forward=True, overlap_ratio=0.2):
        """
        Scroll through the EEG timeline with overlap.

        overlap_ratio: fraction of the window to overlap
            0.0 → no overlap (step = full window)
            0.5 → 50% overlap (step = half window advance)
            0.8 → 80% overlap (step = 20% advance)
        """
        step = self.time_window * (1 - overlap_ratio)

        if forward:
            if self.current_time + step < self.duration:
                self.current_time += step
        else:
            self.current_time = max(0, self.current_time - step)

        self.plot_current_window()


    def set_time_window(self, seconds):
        self.time_window = seconds
        self.current_time = 0
        self.plot_current_window()

    def update_channels(self, selected):
        self.selected_channels = selected
        self.plot_current_window()

    def get_axes_screen_coords(self, ax, fig):
        # Get the axes position in figure coordinates
        bbox = ax.get_position()
        # Bottom-left and top-right corners in figure coordinates
        x0, y0 = bbox.x0, bbox.y0  # Bottom-left
        x1, y1 = bbox.x1, bbox.y1  # Top-right
        # Transform to display (screen) coordinates
        bottom_left = fig.transFigure.transform([x0, y0])
        top_right = fig.transFigure.transform([x1, y1])
        # Return xmin, xmax, ymin, ymax
        return [bottom_left[0], top_right[0], bottom_left[1], top_right[1]]
    
    def is_point_in_bounds(self, x, y, xmin, xmax, ymin, ymax):
        # Check if the point (x, y) falls within the given bounds
        return xmin <= x <= xmax and ymin <= y <= ymax