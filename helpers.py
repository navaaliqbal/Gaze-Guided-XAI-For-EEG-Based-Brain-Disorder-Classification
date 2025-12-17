import cv2
import numpy as np
from PyQt5 import QtGui

def qpixmap_to_cv(qpixmap):
    qimg = qpixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.array(ptr).reshape(height, width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)