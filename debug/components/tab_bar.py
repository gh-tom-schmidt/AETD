from PySide6.QtWidgets import QTabWidget, QWidget, QHBoxLayout, QPushButton
from .tabs import NavTab, SpeedTab, RoadObjectTab, SegmentorTab, PathTab


class ModuleTabBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # layout to hold both the tab bar and the button
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # save all tabs
        self.tabs = []

        # hook tabs
        tab = NavTab()
        self.tab_widget.addTab(tab, "Navigation Data Extractor")
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

        # add self here to get already calculated segments from the image viewer
        tab = PathTab(self)
        self.tab_widget.addTab(tab, "Path Planner")
        self.tabs.append(tab)

        # "run all modules at once" button
        run_button = QPushButton("Run All")
        run_button.clicked.connect(self.run_all_processes)
        layout.addWidget(run_button)

    def run_all_processes(self):
        for tab in self.tabs:
            tab.process()
