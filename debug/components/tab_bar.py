from PySide6.QtWidgets import QHBoxLayout, QTabWidget, QWidget

from .module_tab import ModulTab
from .tabs import DirectionsTab, FullAnnotationTab, PathTab, RoadObjectTab, SegmentorTab, SpeedTab


class ModuleTabBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        # layout to hold both the tab bar and the button
        layout = QHBoxLayout(parent=self)
        layout.setContentsMargins(left=0, top=0, right=0, bottom=0)

        self.tab_widget = QTabWidget()
        layout.addWidget(arg__1=self.tab_widget)

        # save all tabs
        self.tabs: list[ModulTab] = []

        # hook tabs
        tab = DirectionsTab()
        self.tab_widget.addTab(widget=tab, arg__2="Direction Data Extractor")
        self.tabs.append(tab)

        tab = SpeedTab()
        self.tab_widget.addTab(widget=tab, arg__2="Speed Data Extractor")
        self.tabs.append(tab)

        tab = RoadObjectTab()
        self.tab_widget.addTab(widget=tab, arg__2="Road Object Detector")
        self.tabs.append(tab)

        tab = SegmentorTab()
        self.tab_widget.addTab(widget=tab, arg__2="Road Segmentor")
        self.tabs.append(tab)

        tab = PathTab()
        self.tab_widget.addTab(widget=tab, arg__2="Path Planner")
        self.tabs.append(tab)

        tab = FullAnnotationTab()
        self.tab_widget.addTab(widget=tab, arg__2="Full Annotation Pipeline")
        self.tabs.append(tab)
