import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.ndimage import gaussian_filter

class EEGViewer:
    def __init__(self, edf_path, fixation_json_path, time_window=5.0, overlap_ratio=0.2,
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

        # ==== Load Fixations ====
        with open(fixation_json_path, "r") as f:
            data = json.load(f)

        # ✅ Flatten all fixations from all sessions
        self.fixations = []
        if "sessions" in data:
            for session in data["sessions"]:
                if "fixations" in session:
                    for fx in session["fixations"]:
                        fx["session_id"] = session.get("session_id", "unknown")
                        self.fixations.append(fx)
        elif "fixations" in data:
            self.fixations = data["fixations"]

        print(f"\n=== FIXATION INFO ===")
        print(f"Loaded {len(self.fixations)} total fixations across sessions")

        if self.fixations:
            # Print first few fixations
            for i, fx in enumerate(self.fixations[:5]):
                print(f"  Fixation {i} | Session: {fx.get('session_id','N/A')}")
                print(f"    x: {fx.get('x','N/A')}  y: {fx.get('y','N/A')}  duration: {fx.get('duration','N/A')}")
                print(f"    channel: {fx.get('channel','N/A')}")
                print(f"    start_time: {fx.get('start_time','N/A')}  end_time: {fx.get('end_time','N/A')}")

        # If fixations have timestamps, set initial time window
        x_values = [fx["x"] for fx in self.fixations if fx.get("x") is not None]
        if x_values:
            self.current_time = max(0, min(x_values) - time_window / 2)
        else:
            self.current_time = 0.0

        # ==== Parameters ====
        self.time_window = time_window
        self.overlap_ratio = overlap_ratio
        self.amplitude_scale = amplitude_scale
        self.offset_step = offset_step
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_alpha = heatmap_alpha

        # ==== Matplotlib Figure ====
        plt.ion()
        self.fig, (self.ax, self.cax) = plt.subplots(1, 2, figsize=(16, 7),
                                                     gridspec_kw={'width_ratios': [15, 1]})
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cbar = None
        self.plot_current_window()
        plt.show(block=True)

    def get_fixations_in_current_window(self):
        """Return all fixations that occur within the current time window"""
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

        print(f"  -> Found {len(current_fixations)} fixations in current window [{window_start:.2f}-{window_end:.2f}]")
        return current_fixations

    def create_heatmap_for_window(self, current_fixations, channel_offsets, window_width, window_height):
        """Create a heatmap for the current window's fixations"""
        if not current_fixations:
            return None

        grid_w, grid_h = 200, 100
        heatmap = np.zeros((grid_h, grid_w), dtype=float)

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

    def plot_current_window(self):
        self.ax.clear()
        if self.cbar is not None:
            self.cax.clear()

        self.current_time = max(0, min(self.current_time, self.duration - self.time_window))
        start_idx = int(self.current_time * self.sample_rate)
        end_idx = int((self.current_time + self.time_window) * self.sample_rate)
        end_idx = min(end_idx, len(self.signals[0]))

        print(f"\n=== Plotting window [{self.current_time:.2f}-{self.current_time+self.time_window:.2f}s] ===")

        # === Plot EEG ===
        channel_offsets = {}
        window_height = self.offset_step * len(self.labels)
        for i, label in enumerate(self.labels):
            seg = self.signals[i][start_idx:end_idx]
            if len(seg) == 0:
                continue
            times = np.linspace(self.current_time, self.current_time + self.time_window, len(seg))
            offset = i * self.offset_step
            channel_offsets[label] = offset
            self.ax.plot(times, seg * self.amplitude_scale + offset, color='black', linewidth=0.8, zorder=3)
            self.ax.text(times[0] + 0.05, offset, label, va='center', ha='left',
                         fontsize=8, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.9), zorder=4)

        # === Plot Heatmap ===
        current_fixations = self.get_fixations_in_current_window()
        heatmap = self.create_heatmap_for_window(current_fixations, channel_offsets,
                                                 self.time_window, window_height)

        if heatmap is not None:
            extent = [self.current_time, self.current_time + self.time_window, 0, window_height]
            im = self.ax.imshow(heatmap, extent=extent, origin='lower', cmap='hot',
                                alpha=self.heatmap_alpha, aspect='auto', zorder=2)
            if self.cbar is None:
                self.cbar = self.fig.colorbar(im, cax=self.cax)
                self.cbar.set_label('Fixation Intensity', fontsize=9)

        self.ax.set_xlim(self.current_time, self.current_time + self.time_window)
        self.ax.set_ylim(-100, window_height)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title(
            f"EEG + Fixation Heatmap | Time {self.current_time:.1f}-{self.current_time+self.time_window:.1f}s | {len(current_fixations)} fixations",
            fontsize=12, weight='bold'
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def scroll_time(self, forward=True):
        step = self.time_window * (1 - self.overlap_ratio)
        if forward:
            self.current_time = min(self.duration - self.time_window, self.current_time + step)
        else:
            self.current_time = max(0, self.current_time - step)
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
            self.current_time = self.duration - self.time_window
            self.plot_current_window()

if __name__ == "__main__":
    EEGViewer(
        edf_path=r"C:\Users\S.S.T\Downloads\0001146.edf",
        fixation_json_path=r"C:\Users\S.S.T\Documents\VsCode\Gaze Data Collection\64\output_fixations_all_sessions_new.json",
        time_window=5.0,
        overlap_ratio=0.2,
        heatmap_sigma=8,
        heatmap_alpha=0.7
    )
