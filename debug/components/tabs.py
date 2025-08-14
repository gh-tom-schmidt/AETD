from modules import (
    NavigationDataExtractor,
    SpeedDataExtractor,
    RoadObjectDetector,
    RoadSegmentor,
    PathPlanner,
)
from .module_tab import ModulTab
from configs import (
    CLASSIFICATION_MODEL_PATH,
    SEGMENTATION_MODEL_PATH,
    DETECTION_MODEL_PATH,
)


class NavTab(ModulTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.modul = NavigationDataExtractor()


class SpeedTab(ModulTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.modul = SpeedDataExtractor()


class RoadObjectTab(ModulTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.modul = RoadObjectDetector(DETECTION_MODEL_PATH, CLASSIFICATION_MODEL_PATH)


class SegmentorTab(ModulTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.modul = RoadSegmentor(SEGMENTATION_MODEL_PATH)


# the path planner needs previously calculated segments to work
class PathTab(ModulTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.modul = PathPlanner()
