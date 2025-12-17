# py -3.10 main.py

import sys
import signal
import os
from PyQt5 import QtWidgets
# from main_window import MainWindow
from main_window_NMT import MainWindow
# from main_window_tuab import MainWindow

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    os.path.dirname(__file__),
    ".venv/lib/python3.12/site-packages/PyQt5/Qt5/plugins"
)

def handle_sigint(sig, frame):
    print("\nGracefully shutting down...")
    QtWidgets.QApplication.quit()

def main():
    signal.signal(signal.SIGINT, handle_sigint)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()