import cv2
from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal
import numpy as np


class ImageViewer(QLabel):
    # emit the image to all modules (tabs)
    image_ready = Signal(np.ndarray)

    def __init__(self, img_path=None, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.img_path = img_path

        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        if self.img_path:
            self.img = cv2.imread(self.img_path)
            # emit the image to all modules (tabs)
            self.image_ready.emit(self.img)
            self.redraw()

    def redraw(self):
        if self.img is None:
            return

        rgb_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        if self.img is not None:
            self.redraw()
        super().resizeEvent(event)

    def handle_result(self, results):
        print(results)
