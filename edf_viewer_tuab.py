# navaal
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

class EDFPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.fig.set_size_inches(10, 4)
        self.ax = self.fig.add_subplot(111)
        # self.ax.set_aspect('equal')
        super().__init__(self.fig)
        self.setParent(parent)
        self.signals = []
        self.labels = []
        self.sample_rate = 0
        self.current_time = 0
        self.time_window = 5
        self.selected_channels = []
        self.amplitude_scale = 100
        

        self.heatmap_loaded = False
        self.heatmap_file_path = None
        self.heatmap_synthetic = False

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

    def load_heatmap(self, file_path=None, synthetic=False):
        self.heatmap_loaded = True
        self.heatmap_file_path = file_path
        self.heatmap_synthetic = synthetic
        self.draw_heatmap()

    def draw_heatmap(self):
        raw_gaze_data = []
        if self.heatmap_file_path is not None and not self.heatmap_synthetic:
            with open(self.heatmap_file_path, "r") as f:
                data = json.load(f)
                for s in data["session"]:
                    if s["time_window"] == self.time_window and s["time_window_start"] == self.current_time:
                        raw_gaze_data += s["gaze_data"]

        if self.heatmap_synthetic:
            with open('recordings/synthetic.json', "r") as f:
                data = json.load(f)
                raw_gaze_data += data["gaze_data"]

        # Use the same coordinate system for both scatter and heatmap:
        transformed_points = []
        for r in raw_gaze_data:
            # Convert the point from screen (or original) coordinates to axes data coordinates
            inv_transform = self.ax.transData.inverted()
            axes_coords = inv_transform.transform((r['x'], r['y']))
            r['x'] = axes_coords[0]
            r['y'] = axes_coords[1]
            transformed_points.append((r['x'], r['y']))

        # Extract separate arrays for the KDE and plot
        transformed_points = np.array(transformed_points)
        if transformed_points.size == 0:
            return  # No points to plot

        raw_gaze_data_x = transformed_points[:, 0]
        raw_gaze_data_y = transformed_points[:, 1]

        # Optionally, check current axes limits or define your own grid limits.
        # Using the current axes limits ensures consistency.
        x_left, x_right = self.ax.get_xlim()
        y_bottom, y_top = self.ax.get_ylim()

        # Build a grid covering the same data range as the axes
        grid_x, grid_y = np.mgrid[x_left:x_right:1000j, y_bottom:y_top:1000j]
        grid_positions = np.vstack([grid_x.ravel(), grid_y.ravel()])

        # Compute the KDE with the transformed data points
        positions = np.vstack([raw_gaze_data_x, raw_gaze_data_y])
        kde = gaussian_kde(positions, bw_method=0.1)
        z = np.reshape(kde(grid_positions).T, grid_x.shape)

        # Plot the heatmap over the axes using the same extent as the axes data coordinates
        self.ax.imshow(
            z.T,
            extent=[x_left, x_right, y_bottom, y_top],
            origin='lower',
            alpha=0.5,
            cmap='viridis',
            aspect='auto',
            zorder=10,
        )
        
        # Plot your transformed scatter points (they're already in data coordinates)
        # self.ax.plot(raw_gaze_data_x, raw_gaze_data_y, 'ro', scalex=False)
        # self.ax.scatter(raw_gaze_data_x, raw_gaze_data_y)
        
        self.draw()
    
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


    def plot_current_window(self, debug_colors=False):
        if not self.signals:
            return

        self.ax.clear()
        start_idx = int(self.current_time * self.sample_rate)
        end_idx = int((self.current_time + self.time_window) * self.sample_rate)

        offset_step = 200  # Fixed vertical offset for stacked display
        visible_channels = [label for label in self.labels if label in self.selected_channels]
        num_visible = len(visible_channels)

        self.offset_step = offset_step
        self.num_visible = len(visible_channels)

        # --- Plot EEG traces (black or RGB depending on debug_colors) ---
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
                self.ax.plot(times, segment + offset, color=rgb, linewidth=1.5)

                label_x = self.current_time + self.time_window * 0.01
                self.ax.text(
                    label_x, offset, label,
                    va='center', ha='left', fontsize=9, fontweight='bold',
                    color=rgb,
                    bbox=dict(facecolor='white', alpha=0.9)
                )
            else:
                self.ax.plot(times, segment + offset, color='black')
                label_x = self.current_time + self.time_window * 0.01
                self.ax.text(
                    label_x, offset, label,
                    va='center', ha='left', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9)
                )

        # --- Scale bar (50 µV, 1s) ---
        bar_x = self.current_time + self.time_window - 1.5
        bar_y = -100
        self.ax.plot([bar_x, bar_x], [bar_y, bar_y + 50], color='black', lw=2)
        self.ax.plot([bar_x, bar_x + 1], [bar_y, bar_y], color='black', lw=2)
        self.ax.text(bar_x + 0.1, bar_y + 25, "50 µV", va='center')
        self.ax.text(bar_x + 0.5, bar_y - 10, "1 s", ha='center')

        # --- Time ticks ---
        num_ticks = 10
        time_ticks = np.linspace(self.current_time, self.current_time + self.time_window, num_ticks)
        time_labels = [f"{tick:.2f}" for tick in time_ticks]
        self.ax.set_xticks(time_ticks)
        self.ax.set_xticklabels(time_labels)

        # --- Limits ---
        self.ax.set_xlim(self.current_time, self.current_time + self.time_window)
        self.ax.set_ylim(-150, offset_step * max(1, num_visible))

        # Hide y-axis, keep x-axis
        # Hide the y-axis and only show the x-axis
        self.ax.axis('on')  # Ensure x-axis is visible
        self.ax.get_yaxis().set_visible(False)  # Hide the y-axis

        self.ax.set_title("Clinical-style EEG Viewer", fontsize=12)

        # --- Draw main EEG plot ---
        self.draw()
        if self.heatmap_loaded:
            self.draw_heatmap()

        # --- Build hidden mask buffer for gaze lookup ---
        self.mask_buffer = self.make_channel_mask(start_idx, end_idx, visible_channels, offset_step)

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