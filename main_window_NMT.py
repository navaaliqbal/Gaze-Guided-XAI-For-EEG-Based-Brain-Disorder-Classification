import sys
import os
import threading
import csv
import time
import datetime
import cv2
import numpy as np
import uuid
import json
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QFileDialog, QMessageBox,
    QInputDialog, QApplication, QLabel, QDialog, QVBoxLayout, QPushButton,QWidget, QSpacerItem, QSizePolicy
)
import queue, threading, time
from gaze_tracking import GazeTrackingThread
from gaze_overlay import GazeOverlay
from edf_viewer_NMT import EDFPlotCanvas, ChannelSelectorDialog
from PyQt5.QtGui import QFont
from helpers import qpixmap_to_cv
from collections import deque
import time
# import the thread class (if in separate file)
from speech_thread import SpeechRecognitionThread
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import json
from vosk import Model, KaldiRecognizer

RECORDING_FPS = 30
class MetadataDialog(QDialog):
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        
        # Set minimum and initial size larger
        
        self.setMinimumSize(500, 180)
        self.resize(350, 200)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)  # Center all items vertically and horizontally

        # Add some vertical spacing at top
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Font for labels
        label_font = QFont("Arial", 10)

        # Add metadata labels
        for key, value in metadata.items():
            label = QLabel(f"{key}: {value}")
            label.setAlignment(Qt.AlignCenter)
            label.setFont(label_font)
            main_layout.addWidget(label)

        # Add spacer between labels and button
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # OK button
        ok_btn = QPushButton("OK")
        ok_btn.setFont(QFont("Arial", 10))
        ok_btn.setFixedWidth(100)
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        ok_btn.setAutoDefault(True)
        
        # Center button
        button_layout = QVBoxLayout()
        button_layout.addWidget(ok_btn, alignment=Qt.AlignCenter)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.setModal(True)
        self.adjustSize()

        # Center the dialog on parent
        if parent:
            parent_rect = parent.frameGeometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fullscreen = True
        self.default_width = 1920
        self.default_height = 1080
        self.setWindowTitle("Doctor-Friendly EDF Viewer")
        self.canvas = EDFPlotCanvas(self)
        # self.canvas.fig.set_dpi(300)
        self.setCentralWidget(self.canvas)
        self.heatmap_parameters_dialog = None
        self.gaze_overlay = GazeOverlay(self)
        self.gaze_overlay.toggle_gaze(False)
        self.gaze_overlay.setGeometry(0, 0, self.default_width, self.default_height)
        self.gaze_overlay.raise_()
        self.create_menu()
        self.apply_window_mode()
        self.gaze_thread = None
        self.gaze_tracking_active = False
        self.slice_log = []
        self.current_slice_gaze = []
        self.last_slice_time = None
        
        self.coord_label = QLabel(self)
        self.coord_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.canvas.ax.set_xlim(0, 1)
        self.canvas.ax.set_ylim(0, 1)
        # self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.status_bar = self.statusBar()
        self.canvas_mouse_x = 0
        self.canvas_mouse_y = 0
        self.mouse_raw_x = 0
        self.mouse_raw_y = 0
        self.gaze_x = 0
        self.gaze_y = 0

        # Session recording attributes
        self.session_name = None
        self.session_recording_active = False
        self.session_data = []
        self.current_plot_uid = None
        self.current_gaze_data = []
        self.current_plot_start_time = None
        self.np_data_dir = "np_data"  # Directory where .npy files will be saved
        self.speech_thread = None
        self.gaze_history = deque(maxlen=10000)   # keeps recent gaze points (trim to memory you want)
        self.last_gaze = None                     # latest gaze snapshot

        # buttons
        self.create_toolbar_buttons()
        self.setFocusPolicy(Qt.StrongFocus)



        os.makedirs(self.np_data_dir, exist_ok=True)

    def on_mouse_move(self, event):
        if event.inaxes:
            self.mouse_x, self.mouse_y = event.xdata, event.ydata
            self.mouse_raw_x, self.mouse_raw_y = event.x, event.y
            self.canvas.ax.plot(self.mouse_x, self.mouse_y, 'ro')
            self.canvas.draw()
            self.status_bar.showMessage(self.get_status_bar_message())

    def get_status_bar_message(self):
        return f"Canvas: ({self.mouse_x:.2f}, {self.mouse_y:.2f}), Mouse Raw: ({self.mouse_raw_x:.2f}, {self.mouse_raw_y:.2f}) ({self.gaze_x:.2f}, {self.gaze_y:.2f})"
    def create_toolbar_buttons(self):
        toolbar_widget = QWidget(self)
        toolbar_layout = QtWidgets.QHBoxLayout()
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        toolbar_layout.setSpacing(15)

        # --- Start button (green) ---
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 5px 15px;")
        self.start_btn.clicked.connect(self.handle_start)

        # --- Pause button (orange) ---
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold; padding: 5px 15px;")
        self.pause_btn.clicked.connect(self.handle_pause)

        # --- Resume button (blue) ---
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 5px 15px;")
        self.resume_btn.clicked.connect(self.handle_resume)

        # --- Stop button (red) ---
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 5px 15px;")
        self.stop_btn.clicked.connect(self.handle_stop)

        for b in [self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn]:
            toolbar_layout.addWidget(b)

        toolbar_widget.setLayout(toolbar_layout)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self._make_toolbar(toolbar_widget))
        for b in [self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn]:
            b.setFocusPolicy(Qt.NoFocus)


    def _make_toolbar(self, widget):
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setStyleSheet("QToolBar { background: #f0f0f0; border: none; }")
        tb.addWidget(widget)
        return tb


    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_eeg_heatmap_action = QAction("Load EEG with Heatmap", self)
        load_eeg_heatmap_action.triggered.connect(self.open_eeg_with_heatmap)
        file_menu.addAction(load_eeg_heatmap_action)
        open_action = QAction("Open EDF File", self)
        open_action.triggered.connect(self.open_edf)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        settings_menu = menubar.addMenu("Settings")
        toggle_fullscreen_action = QAction("Toggle Fullscreen", self)
        toggle_fullscreen_action.triggered.connect(self.toggle_fullscreen)
        settings_menu.addAction(toggle_fullscreen_action)

        resize_action = QAction("Set Window Size", self)
        resize_action.triggered.connect(self.set_window_size)
        settings_menu.addAction(resize_action)

        time_action = QAction("Set Time Window", self)
        time_action.triggered.connect(self.set_time_window)
        settings_menu.addAction(time_action)

        channel_action = QAction("Select Channels", self)
        channel_action.triggered.connect(self.select_channels)
        settings_menu.addAction(channel_action)

        self.toggle_gaze_action = QAction("Start Gaze Tracking", self)
        self.toggle_gaze_action.setCheckable(True)
        self.toggle_gaze_action.triggered.connect(self.toggle_gaze_tracking)
        settings_menu.addAction(self.toggle_gaze_action)

        # Rename the existing heatmap action for clarity
        self.load_heatmap_action = QAction("Load Heatmap Only", self)  # Changed text
        self.load_heatmap_action.triggered.connect(self.open_heatmap)
        file_menu.addAction(self.load_heatmap_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.toggle_session_recording_action = QtWidgets.QAction("Start Recording", self)
        self.toggle_session_recording_action.setCheckable(True)
        self.toggle_session_recording_action.triggered.connect(self.toggle_session_recording)
        settings_menu.addAction(self.toggle_session_recording_action)
        # self.toggle_speech_action = QAction("Start Speech Tracking", self)
        # self.toggle_speech_action.setCheckable(True)
        # self.toggle_speech_action.triggered.connect(self.toggle_speech_tracking)
        # settings_menu.addAction(self.toggle_speech_action)


        # self.store_coords = QtWidgets.QAction("Start Recording", self)
        # self.toggle_session_recording_action.setCheckable(True)
        # self.toggle_session_recording_action.triggered.connect(self.toggle_session_recording)
        # settings_menu.addAction(self.toggle_session_recording_action)
        heatmap_menu = menubar.addMenu("Heatmap")
        
        load_heatmap_action = QAction("Load Heatmap Data", self)
        load_heatmap_action.triggered.connect(self.open_heatmap_data)
        heatmap_menu.addAction(load_heatmap_action)
        
        toggle_heatmap_action = QAction("Toggle Heatmap", self)
        toggle_heatmap_action.triggered.connect(self.toggle_heatmap_display)
        heatmap_menu.addAction(toggle_heatmap_action)
        
        heatmap_settings_action = QAction("Heatmap Settings", self)
        heatmap_settings_action.triggered.connect(self.show_heatmap_settings)
        heatmap_menu.addAction(heatmap_settings_action)
    def open_eeg_with_heatmap(self):
        """Open both EDF file and corresponding fixation file together"""
        # First, open EDF file
        edf_path, _ = QFileDialog.getOpenFileName(
            self, "Open EDF File", "", "EDF Files (*.edf);;All Files (*)"
        )
        if not edf_path:
            return
        
        # Then, open fixation file
        fixation_path, _ = QFileDialog.getOpenFileName(
            self, "Open Fixation Data File", "", "JSON Files (*.json);;All Files (*)"
        )
        if not fixation_path:
            return
        
        # Load both files
        self.current_edf_file = os.path.basename(edf_path)
        
        # Load EDF first
        self.canvas.load_edf(edf_path)
        
        # Then load heatmap data
        self.canvas.load_heatmap_data(fixation_path)
        
        # Force refresh to show both
        self.canvas.plot_current_window()
        
        print(f"Loaded EEG: {edf_path}")
        print(f"Loaded fixations: {fixation_path}")
        print("Heatmap should now be visible behind EEG traces")

    def open_heatmap_data(self):
        """Open dialog to load heatmap/fixation data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Heatmap Data File", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.canvas.load_heatmap_data(file_path)
            # Ensure heatmap is enabled and refresh
            self.canvas.show_heatmap = True
            self.canvas.plot_current_window()
            print(f"Heatmap data loaded from: {file_path}")

    def toggle_heatmap_display(self):
        """Toggle heatmap on/off"""
        self.canvas.toggle_heatmap()

    def show_heatmap_settings(self):
        """Show dialog for heatmap parameters"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Heatmap Settings")
        layout = QVBoxLayout()
        
        # Sigma parameter
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_label = QtWidgets.QLabel("Smoothing Sigma:")
        sigma_spin = QtWidgets.QSpinBox()
        sigma_spin.setRange(1, 50)
        sigma_spin.setValue(self.canvas.heatmap_sigma)
        sigma_layout.addWidget(sigma_label)
        sigma_layout.addWidget(sigma_spin)
        
        # Alpha parameter
        alpha_layout = QtWidgets.QHBoxLayout()
        alpha_label = QtWidgets.QLabel("Transparency Alpha:")
        alpha_spin = QtWidgets.QDoubleSpinBox()
        alpha_spin.setRange(0.1, 1.0)
        alpha_spin.setSingleStep(0.1)
        alpha_spin.setValue(self.canvas.heatmap_alpha)
        alpha_layout.addWidget(alpha_label)
        alpha_layout.addWidget(alpha_spin)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        
        def apply_settings():
            self.canvas.set_heatmap_parameters(
                sigma=sigma_spin.value(),
                alpha=alpha_spin.value()
            )
            dialog.accept()
        
        apply_btn.clicked.connect(apply_settings)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(sigma_layout)
        layout.addLayout(alpha_layout)
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec_()






    # def process_speech(self, word_payload):
    #     """
    #     word_payload: { 'word': 'seizure', 'start': abs_start, 'end': abs_end, 'conf': ... }
    #     """
    #     start = word_payload["start"]
    #     end   = word_payload["end"]
    #     word  = word_payload["word"]
    #     conf  = word_payload.get("conf")

    #     # find gaze points whose timestamp is within [start, end]
    #     gaze_window = [g for g in list(self.gaze_history) if g["timestamp"] >= start and g["timestamp"] <= end]

    #     # # fallback: use the latest gaze before the word start
    #     # if not gaze_window:
    #     #     # find last gaze <= start
    #     #     gaze_before = None
    #     #     for g in reversed(self.gaze_history):
    #     #         if g["timestamp"] <= start:
    #     #             gaze_before = g
    #     #             break
    #     #     gaze_window = [gaze_before] if gaze_before else []

    #     # speech_event = {
    #     #     "timestamp": start,
    #     #     "speech": {
    #     #         "word": word_payload["word"],
    #     #         "start": start,
    #     #         "end": end,
    #     #         "conf": word_payload.get("conf")
    #     #     },
    #     #     "gaze_window": gaze_window  # list (possibly empty or single element)
    #     # }

    #     # # add to session storage (and/or print/log)
    #     if not gaze_window and self.gaze_history:
    #         for g in reversed(self.gaze_history):
    #             if g["timestamp"] <= start:
    #                 gaze_window = [g]
    #                 break

    #     # Update gaze points with speech info
    #     for g in gaze_window:
    #         g["word"] = word
    #         g["speech_meta"] = {
    #             "start": start,
    #             "end": end,
    #             "conf": conf
    #         }

    #     # If no matching gaze, create standalone speech-only entry
    #     if not gaze_window:
    #         speech_event = {
    #             "timestamp": start,
    #             "channel": None,
    #             "coords": None,
    #             "raw": None,
    #             "word": word,
    #             "speech_meta": {
    #                 "start": start,
    #                 "end": end,
    #                 "conf": conf
    #             }
    #         }
    #         self.current_gaze_data.append(speech_event)

    #     if self.session_recording_active:
    #         print("Speech aligned:", word, "→", [g["timestamp"] for g in gaze_window])
    #     # print("Speech+Gaze event:", speech_event)
    #     # if self.session_recording_active:
    #     #     # We store in current_gaze_data so session JSON contains both gaze-only and speech+gaze items.
    #     #     self.current_gaze_data.append(speech_event)





    def open_heatmap(self):
        """Load heatmap data only (when EEG is already loaded)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Heatmap JSON File", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.canvas.load_heatmap_data(file_path)
            # Ensure the plot refreshes to show the heatmap
            self.canvas.plot_current_window()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.apply_window_mode()

    def set_window_size(self):
        if self.fullscreen:
            QMessageBox.information(self, "Resize Window", "Please exit fullscreen first.")
            return
        width, ok1 = QInputDialog.getInt(self, "Set Width", "Width:", self.default_width, 800, 3840)
        if not ok1:
            return
        height, ok2 = QInputDialog.getInt(self, "Set Height", "Height:", self.default_height, 600, 2160)
        if ok2:
            self.resize(width, height)
            self.gaze_overlay.setGeometry(0, 0, width, height)

    def set_time_window(self):
        seconds, ok = QInputDialog.getInt(self, "Time Slice (s)", "Seconds per screen:", self.canvas.time_window, 1, 60)
        if ok:
            self.canvas.set_time_window(seconds)

    def select_channels(self):
        if not self.canvas.labels:
            QMessageBox.information(self, "No Data", "Please load an EDF file first.")
            return
        dialog = ChannelSelectorDialog(self.canvas.labels, self.canvas.selected_channels)
        if dialog.exec_():
            selected = dialog.get_selected()
            self.canvas.update_channels(selected)

    def apply_window_mode(self):
        if self.fullscreen:
            self.setWindowFlag(QtCore.Qt.FramelessWindowHint, True)
            self.showFullScreen()
        else:
            self.setWindowFlag(QtCore.Qt.FramelessWindowHint, False)
            self.showNormal()
            self.resize(self.default_width, self.default_height)
        self.gaze_overlay.setGeometry(0, 0, self.width(), self.height())

    def resizeEvent(self, event):
        self.gaze_overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    # def toggle_speech_tracking(self, checked):
    #     if checked:
    #         self.toggle_speech_action.setText("Stop Speech Tracking")
    #         # adjust path to your model folder
    #         model_path = "D:/whispertry/vosk-model-en-in-0.5/vosk-model-en-in-0.5"
    #         self.speech_thread = SpeechRecognitionThread(model_path=model_path)
    #         self.speech_thread.newSpeechWord.connect(self.process_speech)
    #         self.speech_thread.start()
    #         print("Speech thread started.")
    #     else:
    #         self.toggle_speech_action.setText("Start Speech Tracking")
    #         if self.speech_thread is not None:
    #             self.speech_thread.stop()
    #             self.speech_thread.wait(2000)  # wait up to 2s for clean exit
    #             self.speech_thread = None
    #             print("Speech thread stopped.")


    # def toggle_speech_tracking(self, checked):
    #     if checked:
    #         self.toggle_speech_action.setText("Stop Speech Tracking")

    #         model_path = "D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/vosk-model-en-in-0.5/vosk-model-en-in-0.5"

    #         # ensure session_name exists (from save_session_data)
    #         if not hasattr(self, "session_name"):
    #             # fallback if user started tracking before saving session
    #             self.session_name = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")

    #         self.speech_thread = SpeechAndAudioRecorder(
    #             model_path=model_path,
    #             session_name=self.session_name
    #         )
    #         self.speech_thread.newSpeechWord.connect(self.process_speech)
    #         self.speech_thread.start()
    #         print(f"Speech+Audio thread started → {self.session_name}")

    #     else:
    #         self.toggle_speech_action.setText("Start Speech Tracking")
    #         if self.speech_thread is not None:
    #             self.speech_thread.stop()
    #             self.speech_thread.wait(2000)
    #             self.speech_thread = None
    #             print("Speech+Audio thread stopped.")


    def toggle_gaze_tracking(self, checked):
        if checked:
            self.toggle_gaze_action.setText("Stop Gaze Tracking")
            self.gaze_thread = GazeTrackingThread()
            self.gaze_thread.newGazePoint.connect(self.process_gaze)
            self.gaze_thread.start()
            self.gaze_tracking_active = True
            # self.gaze_thread.generate_synthetic_movement()
        else:
            self.toggle_gaze_action.setText("Start Gaze Tracking")
            if self.gaze_thread is not None:
                self.gaze_thread.stop()
                self.gaze_thread = None
            self.gaze_tracking_active = False
    def process_gaze(self, timestamp, x, y):
        self.gaze_overlay.update_gaze(x, y)
        self.gaze_x, self.gaze_y = x, y
        eeg_coords = (None, None)
        channel_label, eeg_value, eeg_time = None, None, None

        # --- 1. Color-mask decoding if available ---
        if hasattr(self.canvas, "mask_buffer") and self.canvas.mask_buffer is not None:
            mask_img = self.canvas.mask_buffer
            h, w, _ = mask_img.shape
            if 0 <= int(y) < h and 0 <= int(x) < w:
                color = tuple(mask_img[int(y), int(x)])
                channel_idx = self.canvas.decode_channel_from_color(color)
                if channel_idx is not None and 0 <= channel_idx < len(self.canvas.selected_channels):
                    channel_label = self.canvas.selected_channels[channel_idx]
        inv_transform = self.canvas.ax.transData.inverted()
        eeg_coords = inv_transform.transform((x, y))
        eeg_time = eeg_coords[0]
        eeg_y = eeg_coords[1]
        # --- 2. Fallback to old axis transform if mask fails ---
        if channel_label is None:
            
            channel_idx = int(round(eeg_y / self.canvas.offset_step))
            if 0 <= channel_idx < self.canvas.num_visible:
                channel_label = self.canvas.selected_channels[channel_idx]

        # --- 3. Lookup EEG value at that time ---
        if channel_label is not None:
            try:
                sample_idx = int(eeg_time * self.canvas.sample_rate)
                chan_full_idx = self.canvas.labels.index(channel_label)
                if 0 <= sample_idx < len(self.canvas.signals[chan_full_idx]):
                    eeg_value = float(self.canvas.signals[chan_full_idx][sample_idx])
            except Exception:
                eeg_value = None

        # --- 4. Save gaze data ---
        gaze_point = {
            "timestamp": timestamp,
            "time": eeg_time,
            "channel": channel_label,
            "value": eeg_value,   # store actual EEG value
            "coords": {"x": eeg_coords[0], "y": eeg_coords[1]},
            "raw": {"x": x, "y": y},
            # "word": None,
            # "speech_meta": None,
        }

        print(f"Gaze {timestamp:.3f}s -> {channel_label}, EEG={eeg_value}")

        self.gaze_history.append(gaze_point)
        self.last_gaze = gaze_point
        if self.session_recording_active:
            self.current_gaze_data.append(gaze_point)
    def capture_plot_data(self):
        if self.session_recording_active:
            self.finalize_current_plot()
            start_idx = int(self.canvas.current_time * self.canvas.sample_rate)
            end_idx = int((self.canvas.current_time + self.canvas.time_window) * self.canvas.sample_rate)

            self.current_gaze_data = []
            self.current_plot_start_time = time.time()

    def finalize_current_plot(self):
        end_time = time.time()
        session_entry = {
            "time_window": self.canvas.time_window,
            "time_window_start": self.canvas.current_time,
            "start_time": self.current_plot_start_time,
            "end_time": end_time,
            "gaze_data": self.current_gaze_data
        }
        self.session_data.append(session_entry)

    def toggle_session_recording(self, checked):
        if checked:
            self.toggle_session_recording_action.setText("Stop Recording")
            self.start_session_recording()
        else:
            self.toggle_session_recording_action.setText("Start Recording")
            self.stop_session_recording()

    def start_session_recording(self):
        self.session_data = []
        self.plot_data_dict = {}
        self.current_plot_uid = None
        self.current_gaze_data = []
        self.session_recording_active = True
        print("Session recording started.")

    def stop_session_recording(self):
        if self.session_recording_active:
            self.finalize_current_plot()
            self.save_session_data()
            self.session_recording_active = False
            print("Session recording stopped.")

    # def save_session_data(self):
    #     if not os.path.exists("recordings"):
    #         os.makedirs("recordings")
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     session_filename = os.path.join("recordings", f"session_{timestamp}.json")
    #     session_data = {
    #         "edf_file": getattr(self, "current_edf_file", None),
    #         "session": self.session_data
    #     }
    #     with open(session_filename, 'w') as f:
    #         json.dump(session_data, f, indent=4)
    #     print(f"Session data saved to {session_filename}")
    def save_session_data(self):
        if not os.path.exists("recordings"):
            os.makedirs("recordings")

        # single canonical session name for this run
        self.session_name = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")

        session_filename = os.path.join("recordings", f"{self.session_name}.json")
        session_data = {
            "edf_file": getattr(self, "current_edf_file", None),
            "session": self.session_data
        }

        with open(session_filename, 'w') as f:
            json.dump(session_data, f, indent=4)

        print(f"Session data saved to {session_filename}")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            if self.session_recording_active:
                self.capture_plot_data()
            self.canvas.scroll_time(forward=True)
        elif event.key() == QtCore.Qt.Key_Left:
            if self.session_recording_active:
                self.capture_plot_data()
            self.canvas.scroll_time(forward=False)
        elif event.key() == QtCore.Qt.Key_Space:
            # Toggle gaze overlay visibility
            self.gaze_overlay.toggle_gaze()
        else:
            super().keyPressEvent(event)


    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.canvas.scroll_time(forward=False)
        else:
            self.canvas.scroll_time(forward=True)
        super().wheelEvent(event)

    def handle_start(self):
        self.start_session_recording()
        
        # --- Start gaze tracking thread ---
        if self.gaze_thread is None or not self.gaze_tracking_active:
            self.gaze_thread = GazeTrackingThread()
            self.gaze_thread.newGazePoint.connect(self.process_gaze)
            self.gaze_thread.start()
            self.gaze_tracking_active = True
            print("Gaze tracking started.")
        
        # # --- Start speech tracking thread ---
        # if self.speech_thread is None:
        #     model_path = "D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/vosk-model-en-in-0.5/vosk-model-en-in-0.5"
        #     self.speech_thread = SpeechAndAudioRecorder(model_path=model_path, session_name=self.session_name)
        #     self.speech_thread.newSpeechWord.connect(self.process_speech)
        #     self.speech_thread.start()
        #     print("Speech tracking started.")
        
        print("All started: session, gaze.")



    def handle_pause(self):
        if self.gaze_thread:
            self.gaze_thread.stop()
            self.gaze_thread = None
            self.gaze_tracking_active = False
            print("Gaze tracking paused.")

        # if self.speech_thread:
        #     self.speech_thread.stop()
        #     self.speech_thread.wait(2000)
        #     self.speech_thread = None
        #     print("Speech tracking paused.")

        print("Session paused — threads stopped, recording continues.")


    def handle_resume(self):
        if not self.session_recording_active:
            print("Cannot resume — no active session to continue.")
            return

        # Only restart gaze tracking if not running
        if self.gaze_thread is None or not self.gaze_tracking_active:
            self.gaze_thread = GazeTrackingThread()
            self.gaze_thread.newGazePoint.connect(self.process_gaze)
            self.gaze_thread.start()
            self.gaze_tracking_active = True
            print("Gaze tracking resumed.")

        # # Only restart speech tracking if not running
        # if self.speech_thread is None:
        #     model_path = "D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/vosk-model-en-in-0.5/vosk-model-en-in-0.5"
        #     self.speech_thread = SpeechAndAudioRecorder(model_path=model_path, session_name=self.session_name)
        #     self.speech_thread.newSpeechWord.connect(self.process_speech)
        #     self.speech_thread.start()
        #     print("Speech tracking resumed.")

        print("Session resumed: continuing gaze tracking.")


    def handle_stop(self):
        if self.gaze_thread:
            self.gaze_thread.stop()
            self.gaze_thread = None
            self.gaze_tracking_active = False

        # if self.speech_thread:
        #     self.speech_thread.stop()
        #     self.speech_thread.wait(2000)
        #     self.speech_thread = None

        if self.session_recording_active:
            self.stop_session_recording()
            print("Session stopped and saved.")


    
    
    def open_edf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EDF File", "", "EDF Files (*.edf);;All Files (*)"
        )
        if not file_path:
            return
        self.current_edf_file = os.path.basename(file_path) 
        # --- Metadata handling (Option 1: match CSV filename to EDF filename) ---
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            base_name_int = str(int(base_name))  # remove leading zeros if present
        except ValueError:
            base_name_int = base_name

        csv_folder = "D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/csv"
        metadata = {}
        csv_path = os.path.join(csv_folder, f"{base_name_int}.csv")

        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                sample = f.read(1024)
                f.seek(0)
                delimiter = "\t" if "\t" in sample else ","
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    metadata["Gender"] = row.get("Gender", row.get("gender", ""))
                    metadata["Age"] = row.get("Age", row.get("age", ""))
                    metadata["Type of Epilepsy"] = ""
                    metadata["Type of Seizures"] = ""
                    metadata["Age of onset of epilepsy"] = ""
                    metadata["Frequency of seizures"] = ""
                    metadata["Last seizure date"] = ""
                    metadata["Family history"] = ""
                    metadata["Other neurological disorders"] = ""
                    metadata["Medications"] = ""
                    metadata["Other treatments"] = ""
                    metadata["Any trigger?"] = ""
                    metadata["Any event observed during EEG?"] = ""
                    metadata["Findings on CT/MRI brain"] = ""
                    metadata["EEG findings"] = ""
                    # metadata["Class"] = row.get("Class", row.get("class", ""))
                    break

        # --- Show Metadata Dialog if info was found ---
        if metadata:
            dialog = MetadataDialog(metadata, parent=self)
            dialog.exec_()

        # --- Load EDF signals ---
        self.canvas.load_edf(file_path)
        self.current_plot_start_time = time.time()

