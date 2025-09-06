from PySide6.QtWidgets import QTabWidget, QWidget

from aetd_modules import AnnotationsContainer

from .module_tabs import (
    DirectionsTab,
    ModulTab,
    PathTab,
    RoadObjectClassificationTab,
    RoadObjectDetectionTab,
    SegmentorTab,
    SpeedTab,
)
from .view_tabs import ImageViewerTab, VideoViewerTab


class ModuleTabBar(QTabWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        # save all tabs
        self.tabs: list[ModulTab] = []

        # hook tabs
        tab = DirectionsTab()
        self.addTab(tab, "Direction Data Extractor")
        self.tabs.append(tab)

        tab = SpeedTab()
        self.addTab(tab, "Speed Data Extractor")
        self.tabs.append(tab)

        tab = RoadObjectDetectionTab()
        self.addTab(tab, "Road Object Detector")
        self.tabs.append(tab)

        tab = RoadObjectClassificationTab()
        self.addTab(tab, "Road Object Classifier")
        self.tabs.append(tab)

        tab = SegmentorTab()
        self.addTab(tab, "Road Segmentor")
        self.tabs.append(tab)

        tab = PathTab()
        self.addTab(tab, "Path Planner")
        self.tabs.append(tab)

    def process(self, annotations_container: AnnotationsContainer) -> AnnotationsContainer:
        for tab in self.tabs:
            annotations_container = tab.process(
                annotations_container=annotations_container, to_none=not tab.check_box.isChecked()
            )
        return annotations_container


class ViewTabBar(QTabWidget):
    def __init__(self, module_tab_bar: ModuleTabBar, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # save all tabs
        self.tabs: list[ImageViewerTab | VideoViewerTab] = []

        # hook tabs
        tab = ImageViewerTab(module_tab_bar=module_tab_bar)
        self.addTab(tab, "Image Viewer")
        self.tabs.append(tab)

        tab = VideoViewerTab(module_tab_bar=module_tab_bar)
        self.addTab(tab, "Video Viewer")
        self.tabs.append(tab)
