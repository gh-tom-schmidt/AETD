# main_window.py
from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from configs import globals

from ..components import InfoTable, ModuleTabBar, ViewTabBar
from .premain_layout import PreloadLayout


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        """
        The main window of the application.
        """
        super().__init__()
        self.setWindowTitle(globals.APP_TITLE)

        # create the main widget
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()

        # create the premain layout
        self.premain_layout = PreloadLayout()
        self.main_layout.addWidget(self.premain_layout)

        # File selector
        self.open_button = QPushButton(text="Open")
        self.open_button.setFixedSize(150, 40)
        self.open_button.clicked.connect(slot=self.build)
        self.main_layout.addWidget(self.open_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # add the layout to the central widget
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # apply global stylesheet
        file = QFile(name="styles/global.qss")
        if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            # convert QByteArray to bytes
            data = bytes(file.readAll())
            text: str = data.decode(encoding="utf-8")
            self.setStyleSheet(styleSheet=text)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Override the closeEvent from the MainWindow
        """
        event.accept()

    def build(self) -> None:
        """
        Load the main layout.
        """

        # set the globals
        self.premain_layout.setGlobals()

        # remove the open button from the layout
        self.main_layout.removeWidget(self.open_button)
        self.open_button.setParent(None)
        self.open_button.deleteLater()

        # remove the premain layout
        self.main_layout.removeWidget(self.premain_layout)
        self.premain_layout.setParent(None)
        self.premain_layout.deleteLater()

        # ------------ Vertical Split screen ------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # right widget
        self.module_tab_bar = ModuleTabBar(parent=self)

        # left widget
        left_vertical_splitter = QSplitter(Qt.Orientation.Vertical)

        # left vertical splitter upper widget
        self.view_tab_bar = ViewTabBar(module_tab_bar=self.module_tab_bar)
        left_vertical_splitter.addWidget(self.view_tab_bar)

        # left vertical splitter lower widget
        self.info_table = InfoTable()
        left_vertical_splitter.addWidget(self.info_table)

        # add the splitter to the main layout
        splitter.addWidget(left_vertical_splitter)
        splitter.addWidget(self.module_tab_bar)
        self.main_layout.addWidget(splitter)

        # Caution: the size of the splitter should be set when the
        # layout is built otherwise it will not work correctly
        splitter.setSizes([self.width() * 2 // 3, self.width() // 3])
        left_vertical_splitter.setSizes([self.height() * 2 // 3, self.height() // 3])
