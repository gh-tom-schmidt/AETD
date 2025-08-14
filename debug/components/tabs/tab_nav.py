from modules import NavigationDataExtractor
from components import ModulTab
from PySide6.QtCore import Signal
import numpy as np


class NavTab(ModulTab):
    def __init__(self, parent=None):
        # emit the resutls
        self.result_ready = Signal(str)

        super().__init__(parent)

        self.nav_extractor = NavigationDataExtractor()

    def receive_image(self, img: np.ndarray):
        self.img = img

    def process(self):
        if self.img is not None:
            results = self.nav_extractor.process(self.img)
            self.result_ready.emit(results)

    def run(self):
        self.thread.start()
