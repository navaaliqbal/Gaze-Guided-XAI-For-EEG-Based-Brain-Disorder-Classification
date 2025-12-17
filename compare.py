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

        # Transform screen coords → EEG axes coords
        inv_transform = self.canvas.ax.transData.inverted()
        eeg_coords = inv_transform.transform((x, y))
        eeg_time = eeg_coords[0]   # in seconds
        eeg_y = eeg_coords[1]      # in stacked channel space

        # Map to channel index + label
        channel_idx = int(round(eeg_y / self.canvas.offset_step))
        if 0 <= channel_idx < self.canvas.num_visible:
            channel_label = self.canvas.selected_channels[channel_idx]
        else:
            channel_label = None
        print(f"Gaze at time {timestamp:.3f}s -> Channel: {channel_label}")
        gaze_point = {
                "timestamp": timestamp,
                "time": eeg_time,
                "channel": channel_label,
                "coords": {"x": eeg_coords[0], "y": eeg_coords[1]},
                "raw": {"x": x, "y": y},  # keep raw screen coords too if useful
                "word": None,
                "speech_meta":None
            }
        self.gaze_history.append(gaze_point)
        self.last_gaze = gaze_point
        if self.session_recording_active:
            
            self.current_gaze_data.append(gaze_point)
    def toggle_speech_tracking(self, checked):
        if checked:
            self.toggle_speech_action.setText("Stop Speech Tracking")
            # adjust path to your model folder
            model_path = "D:/TobiiPro.SDK.Python.Windows_2.1.0.1/64/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"
            self.speech_thread = SpeechRecognitionThread(model_path=model_path)
            self.speech_thread.newSpeechWord.connect(self.process_speech)
            self.speech_thread.start()
            print("Speech thread started.")
        else:
            self.toggle_speech_action.setText("Start Speech Tracking")
            if self.speech_thread is not None:
                self.speech_thread.stop()
                self.speech_thread.wait(2000)  # wait up to 2s for clean exit
                self.speech_thread = None
                print("Speech thread stopped.")


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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_filename = os.path.join("recordings", f"session_{timestamp}.json")
        session_data = {
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
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.canvas.scroll_time(forward=False)
        else:
            self.canvas.scroll_time(forward=True)
        super().wheelEvent(event)


    