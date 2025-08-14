from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from .tabs import NavTab


class ModuleTabBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout for this widget
        right_layout = QVBoxLayout(self)

        # Create tab widget
        tab_widget = QTabWidget()

        # First tab
        tab1 = QWidget()
        layout1 = QVBoxLayout(tab1)
        self.nav_tab = NavTab()
        layout1.addWidget(self.nav_tab)

        # Second tab
        tab2 = QWidget()
        layout2 = QVBoxLayout(tab2)
        # layout2.addWidget( ... your widgets here ... )

        # Add tabs
        tab_widget.addTab(tab1, "Navigation Data Extracotr")
        tab_widget.addTab(tab2, "Tab 2")

        # Add tab widget to layout
        right_layout.addWidget(tab_widget)
