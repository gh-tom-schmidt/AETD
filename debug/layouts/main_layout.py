# main_window.py
from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from configs import globals

from ..components import ImageViewer, ModuleTabBar


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        """
        The main window of the application.
        """
        super().__init__()
        self.setWindowTitle(arg__1=globals.APP_TITLE)

        # create the main widget
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()

        # File selector
        self.open_button = QPushButton(text="Open Image")
        self.open_button.setFixedSize(w=150, h=40)
        self.open_button.clicked.connect(slot=self.selectAndBuild)
        self.main_layout.addWidget(arg__1=self.open_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # add the layout to the central widget
        self.central_widget.setLayout(arg__1=self.main_layout)
        self.setCentralWidget(widget=self.central_widget)

        # apply global stylesheet
        file = QFile(name="styles/global.qss")
        if file.open(flags=QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            # convert QByteArray to bytes
            data = bytes(file.readAll())
            text: str = data.decode(encoding="utf-8")
            self.setStyleSheet(styleSheet=text)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Override the closeEvent from the MainWindow
        """
        event.accept()

    def selectAndBuild(self) -> None:
        """
        Opens a file dialog to select a image file and loads the main layout.
        """

        file_path: str = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Image File",
            dir="",
            filter="Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
        )[0]

        # remove the open button from the layout
        self.main_layout.removeWidget(w=self.open_button)
        self.open_button.setParent(parent=None)
        self.open_button.deleteLater()

        # ------------ Vertical Split screen ------------------
        splitter = QSplitter(arg__1=Qt.Orientation.Horizontal)

        # left widget
        self.image_viewer = ImageViewer(img_path=file_path)
        splitter.addWidget(widget=self.image_viewer)

        # right widget
        self.tab_bar = ModuleTabBar(parent=self)
        splitter.addWidget(widget=self.tab_bar)
        self.main_layout.addWidget(arg__1=splitter)

        # connect tabs from the tabbar and the image viewer
        for tab in self.tab_bar.tabs:
            self.image_viewer.image_ready.connect(slot=tab.receive_annotation_container)
            tab.result_ready.connect(slot=self.image_viewer.update_container)

        self.image_viewer.emit_annotation_container()

        # Caution: the size of the spliter should be set when the
        # layout is built otherwise it will not work correctly
        splitter.setSizes(list=[self.width() * 2 // 3, self.width() // 3])
