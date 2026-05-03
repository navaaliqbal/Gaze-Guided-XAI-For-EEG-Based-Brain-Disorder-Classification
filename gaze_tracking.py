import sys
import socket
import xml.etree.ElementTree as ET
import tobii_research as tr
import time
import numpy as np
from collections import deque
from PyQt5 import QtCore


class GazeTrackingThread(QtCore.QThread):
    newGazePoint = QtCore.pyqtSignal(float, int, int)  # x, y

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        # self.HOST = "100.87.72.127"
        # self.PORT = 4242
        self.SCREEN_WIDTH = 1920
        self.SCREEN_HEIGHT = 1080
        self.WINDOW_SIZE = 5
        self.gaze_history_x = deque(maxlen=self.WINDOW_SIZE)
        self.gaze_history_y = deque(maxlen=self.WINDOW_SIZE)
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 2
        for _ in range(self.WINDOW_SIZE):
            self.gaze_history_x.append(center_x)
            self.gaze_history_y.append(center_y)
        self.circle_radius = 20
        self.correction_offset = 19
        self.eyetracker = None

    def run(self):
        # Step 1: Find eye tracker
        found_eyetrackers = tr.find_all_eyetrackers()
        if not found_eyetrackers:
            print("No eye tracker found. Please connect one and try again.")
            exit()

        self.eyetracker = found_eyetrackers[0]
        print("Connected to:", self.eyetracker.address, self.eyetracker.model)
        print("Address: " + self.eyetracker.address)
        print("Model: " + self.eyetracker.model)
        print("Name (It's OK if this is empty): " + self.eyetracker.device_name)
        print("Serial number: " + self.eyetracker.serial_number)


        # Step 2: Define callback
        def gaze_data_callback(gaze_data):
            # Tobii gives normalized coords (0..1 relative to screen)
            left_eye = gaze_data['left_gaze_point_on_display_area']
            right_eye = gaze_data['right_gaze_point_on_display_area']
            print("Left eye: {0} \t Right eye: {1}".format(
        gaze_data['left_gaze_point_on_display_area'],
        gaze_data['right_gaze_point_on_display_area']))

            # Use average of both eyes if both valid, else whichever is valid
            gaze_x, gaze_y = None, None
            if left_eye and 0 <= left_eye[0] <= 1 and 0 <= left_eye[1] <= 1:
                gaze_x, gaze_y = left_eye
            if right_eye and 0 <= right_eye[0] <= 1 and 0 <= right_eye[1] <= 1:
                # If left already valid, average; else take right
                if gaze_x is not None:
                    gaze_x = (gaze_x + right_eye[0]) / 2
                    gaze_y = (gaze_y + right_eye[1]) / 2
                else:
                    gaze_x, gaze_y = right_eye

            if gaze_x is None or gaze_y is None:
                return  # skip invalid points

            # Convert normalized [0,1] to pixels
            new_gaze_x = int(gaze_x * self.SCREEN_WIDTH)
            new_gaze_y = int(gaze_y * self.SCREEN_HEIGHT)

            # Smooth using deque
            self.gaze_history_x.append(new_gaze_x)
            self.gaze_history_y.append(new_gaze_y)
            smoothed_x = int(sum(self.gaze_history_x) / len(self.gaze_history_x))
            smoothed_y = int(sum(self.gaze_history_y) / len(self.gaze_history_y))

            # Emit to PyQt
            self.newGazePoint.emit(
                time.time(), smoothed_x, self.SCREEN_HEIGHT - smoothed_y - self.correction_offset
            )

        # Step 3: Subscribe
        self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

        # Keep thread alive until stopped
        while self.running:
            self.msleep(10)

        # Step 4: Cleanup
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
        print("Stopped Tobii gaze tracking.")


    def random(self):
        # self.newGazePoint.emit(time.time(), 500, 1080 - 500)
        # self.newGazePoint.emit(time.time(), 0, 0)
        # self.newGazePoint.emit(time.time(), 500, 1080 - 400)
        self.newGazePoint.emit(time.time(), 500, 1080 - 800 - 19)
        # self.newGazePoint.emit(time.time(), 100, 1080 - 800)

    # def generate_synthetic_movement(self, screen_width=1920, screen_height=1080, frequency=0.05, amplitude=200, num_points=100):
    #     x = np.linspace(0, 10000, num_points)
    #     y = screen_height / 2 + amplitude * np.sin(2 * np.pi * frequency * x)
    #     noise_x = np.random.normal(loc=0, scale=10, size=(num_points, 5))
    #     noise_y = np.random.normal(loc=0, scale=10, size=(num_points, 5))
    #     points_x = np.repeat(x, 5) + noise_x.flatten()
    #     points_y = np.repeat(y, 5) + noise_y.flatten()
    #     # print(points_x, points_y)

    #     for i in range(num_points):
    #         # print(points_x[i], points_y[i])
    #         self.newGazePoint.emit(time.time(), int(
    #             points_x[i]), 1080 - int(points_y[i]) - self.correction_offset)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
