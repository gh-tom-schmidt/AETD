from PySide6.QtWidgets import QWidget

from aetd_modules import (
    DirectionExtractor,
    PathPlanner,
    Pipeline,
    RoadObjectExtractor,
    RoadSegmentsExtractor,
    SpeedDataExtractor,
)

from .module_tab import ModulTab


class DirectionsTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul = DirectionExtractor()


class SpeedTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul = SpeedDataExtractor()


class RoadObjectTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul = RoadObjectExtractor()


class SegmentorTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul = RoadSegmentsExtractor()


# the path planner needs previously calculated segments to work
class PathTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul = PathPlanner()


class FullAnnotationTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul = Pipeline()
