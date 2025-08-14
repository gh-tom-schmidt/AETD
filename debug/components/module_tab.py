from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtCore import Signal, Slot
import numpy as np
from modules import PathPlanner


class ModulTab(QWidget):
    # emit the resutls
    result_ready = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img = None
        self.modul = None
        self.segments = None

        # ---------------- LAYOUT ------------------------
        main_layout = QVBoxLayout(self)

        # top row
        top_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.process)
        top_layout.addWidget(self.run_button)
        # pushes everything else to the right
        top_layout.addStretch()

        main_layout.addLayout(top_layout)
        self.setLayout(main_layout)

    @Slot(np.ndarray)
    def receive_image(self, img: np.ndarray):
        self.img = img

    def run(self):
        self.thread.start()

    def process(self):
        if self.img is not None:
            if self.modul is not None:
                # if it is the PathPlanner, than insert the segments
                if isinstance(self.modul, PathPlanner):
                    segments = (
                        self.parent()
                        .parent()
                        .parent()
                        .parent()
                        .parent()
                        .parent()
                        .image_viewer.segments
                    )
                    if segments is not None:
                        results = self.modul.process(segments)

                else:
                    results = self.modul.process(self.img)

                self.result_ready.emit(results)
