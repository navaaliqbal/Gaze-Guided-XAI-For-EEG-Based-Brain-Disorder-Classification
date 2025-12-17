from PyQt5 import QtWidgets, QtCore, QtGui

class GazeOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.gaze_x = 960
        self.gaze_y = 540
        self.circle_radius = 10
        self.points = []
        self.show_gaze = True 

    def toggle_gaze(self, state=None):
        """Toggle or explicitly set visibility of gaze circle."""
        if state is None:
            self.show_gaze = not self.show_gaze
        else:
            self.show_gaze = bool(state)
        self.update()

    def update_gaze(self, x, y):
        # self.points.append((x, 1080 - y))
        # if (len(self.points) > 99):
        #     self.update()
        self.gaze_x = x
        self.gaze_y = 1080 - y
        self.update()

    def paintEvent(self, event):
        if not self.show_gaze:
            return
        painter = QtGui.QPainter(self)
        painter.setOpacity(0.5)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.blue)
        painter.drawEllipse(QtCore.QPoint(self.gaze_x, self.gaze_y),
                            self.circle_radius, self.circle_radius)