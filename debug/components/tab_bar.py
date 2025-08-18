from PySide6.QtWidgets import QHBoxLayout, QTabWidget, QWidget

from .module_tab import ModulTab
from .tabs import DirectionsTab, FullAnnotationTab, PathTab, RoadObjectTab, SegmentorTab, SpeedTab


class ModuleTabBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # layout to hold both the tab bar and the button
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # save all tabs
        self.tabs: list[ModulTab] = []

        # hook tabs
        tab = DirectionsTab()
        self.tab_widget.addTab(tab, "Direction Data Extractor")
        self.tabs.append(tab)

        tab = SpeedTab()
        self.tab_widget.addTab(tab, "Speed Data Extractor")
        self.tabs.append(tab)

        tab = RoadObjectTab()
        self.tab_widget.addTab(tab, "Road Object Detector")
        self.tabs.append(tab)

        tab = SegmentorTab()
        self.tab_widget.addTab(tab, "Road Segmentor")
        self.tabs.append(tab)

        tab = PathTab()
        self.tab_widget.addTab(tab, "Path Planner")
        self.tabs.append(tab)

        tab = FullAnnotationTab()
        self.tab_widget.addTab(tab, "Full Annotation Pipeline")
        self.tabs.append(tab)
