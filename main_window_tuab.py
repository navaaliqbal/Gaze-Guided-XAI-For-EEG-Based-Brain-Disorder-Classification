import sys
import os
from pathlib import Path
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
from gaze_tracking import GazeTrackingThread
from gaze_overlay import GazeOverlay
from edf_viewer_tuab import EDFPlotCanvas, ChannelSelectorDialog
from PyQt5.QtGui import QFont
from helpers import qpixmap_to_cv
from collections import deque
import time
import sounddevice as sd
import soundfile as sf
import json
import threading

from vosk import Model, KaldiRecognizer
# import the thread class (if in separate file)
from speech_thread import SpeechRecognitionThread

RECORDING_FPS = 30
class MetadataDialog(QDialog):
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        
        # Set minimum and initial size larger
        
        self.setMinimumSize(300, 180)
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
        self.session_recording_active = False
        self.session_data = []
        self.current_plot_uid = None
        self.current_gaze_data = []
        self.current_plot_start_time = None
        self.np_data_dir = "np_data"  # Directory where .npy files will be saved
        self.speech_thread = None
        self.gaze_history = deque(maxlen=10000)   # keeps recent gaze points (trim to memory you want)
        self.last_gaze = None                     # latest gaze snapshot

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

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
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

        self.load_heatmap_action = QAction("Load Heatmap", self)
        self.load_heatmap_action.triggered.connect(self.open_heatmap)
        file_menu.addAction(self.load_heatmap_action)

        self.toggle_session_recording_action = QtWidgets.QAction("Start Recording", self)
        self.toggle_session_recording_action.setCheckable(True)
        self.toggle_session_recording_action.triggered.connect(self.toggle_session_recording)
        settings_menu.addAction(self.toggle_session_recording_action)

        self.toggle_speech_action = QAction("Start Speech Tracking", self)
        self.toggle_speech_action.setCheckable(True)
        self.toggle_speech_action.triggered.connect(self.toggle_speech_tracking)
        settings_menu.addAction(self.toggle_speech_action)

        # self.show_metadata_action = QAction("Show Patient Metadata", self)
        # self.show_metadata_action.setCheckable(True)
        # self.show_metadata_action.triggered.connect(self.show_metadata)
        # settings_menu.addAction(self.show_metadata_action)


    def open_edf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EDF File", "", "EDF Files (*.edf);;All Files (*)"
        )
        if not file_path:
            return
        self.current_edf_file = os.path.basename(file_path)

        # Show Metadata if available
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        metadata = {}

        csv_folder = Path("D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/csv")

        if csv_folder.exists():
            for csv_file in csv_folder.glob("*.csv"):
                with open(csv_file, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        csv_name = os.path.splitext(os.path.basename(row.get("original_file_name", "")))[0]
                        if csv_name == base_name:
                            metadata["Age"] = row.get("age", "")
                            metadata["Gender"] = row.get("gender", "")
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
                            break
                if metadata:
                    break

        if metadata:
            dialog = MetadataDialog(metadata, parent=self)
            dialog.exec_()

        # Load EDF signals
        self.canvas.load_edf(file_path)
        self.current_plot_start_time = time.time()

    
    def open_heatmap(self):
        # file_path = 'recordings/synthetic.json'
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Heatmap JSON File", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.canvas.load_heatmap(file_path)

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

    def process_gaze(self, timestamp, x, y):
        self.gaze_overlay.update_gaze(x, y)
        self.gaze_x, self.gaze_y = x, y

        channel_label, eeg_value, eeg_time = None, None, None
        eeg_coords = (None, None)
        # --- 1. Color-mask decoding if available ---
        if hasattr(self.canvas, "mask_buffer") and self.canvas.mask_buffer is not None:
            mask_img = self.canvas.mask_buffer
            h, w, _ = mask_img.shape
            if 0 <= int(y) < h and 0 <= int(x) < w:
                color = tuple(mask_img[int(y), int(x)])
                channel_idx = self.canvas.decode_channel_from_color(color)
                if channel_idx is not None and 0 <= channel_idx < len(self.canvas.selected_channels):
                    channel_label = self.canvas.selected_channels[channel_idx]

        # --- 2. Fallback to old axis transform if mask fails ---
        if channel_label is None:
            inv_transform = self.canvas.ax.transData.inverted()
            eeg_coords = inv_transform.transform((x, y))
            eeg_time = eeg_coords[0]
            eeg_y = eeg_coords[1]
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
            "word": None,
            "speech_meta": None,
        }

        print(f"Gaze {timestamp:.3f}s -> {channel_label}, EEG={eeg_value}")

        self.gaze_history.append(gaze_point)
        self.last_gaze = gaze_point
        if self.session_recording_active:
            self.current_gaze_data.append(gaze_point)
    def toggle_speech_tracking(self, checked):
        if checked:
            self.toggle_speech_action.setText("Stop Speech Tracking")

            model_path = "D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"

            # ensure session_name exists (from save_session_data)
            if not hasattr(self, "session_name"):
                # fallback if user started tracking before saving session
                self.session_name = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")

            self.speech_thread = SpeechAndAudioRecorder(
                model_path=model_path,
                session_name=self.session_name
            )
            self.speech_thread.newSpeechWord.connect(self.process_speech)
            self.speech_thread.start()
            print(f"Speech+Audio thread started → {self.session_name}")

        else:
            self.toggle_speech_action.setText("Start Speech Tracking")
            if self.speech_thread is not None:
                self.speech_thread.stop()
                self.speech_thread.wait(2000)
                self.speech_thread = None
                print("Speech+Audio thread stopped.")




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
    def process_speech(self, word_payload):
        """
        word_payload: { 'word': 'seizure', 'start': abs_start, 'end': abs_end, 'conf': ... }
        """
        start = word_payload["start"]
        end   = word_payload["end"]
        word  = word_payload["word"]
        conf  = word_payload.get("conf")

        # find gaze points whose timestamp is within [start, end]
        gaze_window = [g for g in list(self.gaze_history) if g["timestamp"] >= start and g["timestamp"] <= end]

        # # fallback: use the latest gaze before the word start
        # if not gaze_window:
        #     # find last gaze <= start
        #     gaze_before = None
        #     for g in reversed(self.gaze_history):
        #         if g["timestamp"] <= start:
        #             gaze_before = g
        #             break
        #     gaze_window = [gaze_before] if gaze_before else []

        # speech_event = {
        #     "timestamp": start,
        #     "speech": {
        #         "word": word_payload["word"],
        #         "start": start,
        #         "end": end,
        #         "conf": word_payload.get("conf")
        #     },
        #     "gaze_window": gaze_window  # list (possibly empty or single element)
        # }

        # # add to session storage (and/or print/log)
        if not gaze_window and self.gaze_history:
            for g in reversed(self.gaze_history):
                if g["timestamp"] <= start:
                    gaze_window = [g]
                    break

        # Update gaze points with speech info
        for g in gaze_window:
            g["word"] = word
            g["speech_meta"] = {
                "start": start,
                "end": end,
                "conf": conf
            }

        # If no matching gaze, create standalone speech-only entry
        if not gaze_window:
            speech_event = {
                "timestamp": start,
                "channel": None,
                "coords": None,
                "raw": None,
                "word": word,
                "speech_meta": {
                    "start": start,
                    "end": end,
                    "conf": conf
                }
            }
            self.current_gaze_data.append(speech_event)

        if self.session_recording_active:
            print("Speech aligned:", word, "→", [g["timestamp"] for g in gaze_window])
        # print("Speech+Gaze event:", speech_event)
        # if self.session_recording_active:
        #     # We store in current_gaze_data so session JSON contains both gaze-only and speech+gaze items.
        #     self.current_gaze_data.append(speech_event)

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

class SpeechAndAudioRecorder(QtCore.QThread):
    """
    Handles continuous microphone capture.
    - Saves full .wav file in recordings/
    - Streams audio frames to Vosk recognizer
    - Emits recognized words in near real-time
    """
    newSpeechWord = QtCore.pyqtSignal(dict)  # same payload type you already use

    def __init__(self, model_path, session_name=None, output_dir="recordings", samplerate=16000):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.session_name = session_name or time.strftime("session_%Y%m%d_%H%M%S")
        self.samplerate = samplerate
        self.channels = 1
        self.stop_event = threading.Event()


    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        session_name = time.strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(self.output_dir, f"{self.session_name}.wav")


        model = Model(self.model_path)
        rec = KaldiRecognizer(model, self.samplerate)
        rec.SetWords(True)

        with sf.SoundFile(audio_path, mode='w', samplerate=self.samplerate, channels=self.channels) as wf:
            def callback(indata, frames, time_info, status):
                if status:
                    print("Audio status:", status)
                if self.stop_event.is_set():
                    raise sd.CallbackStop

                wf.write(indata)

                # Vosk transcription
                if rec.AcceptWaveform(indata.tobytes()):
                    result = json.loads(rec.Result())
                    if "result" in result:
                        for w in result["result"]:
                            word_payload = {
                                "word": w["word"],
                                "start": w["start"],
                                "end": w["end"],
                                "conf": w.get("conf", None),
                            }
                            self.newSpeechWord.emit(word_payload)
                # You can also emit partials here if you want

            with sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=callback):
                print(f"Recording + transcription started → {audio_path}")
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        print(f"Stopped. Audio saved → {audio_path}")

    def stop(self):
        self.stop_event.set()