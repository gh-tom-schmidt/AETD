# main_window.py
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QSplitter,
    QTabWidget,
)
from PySide6.QtCore import QFile, Qt
from configs import globals
from ..components import ImageViewer, ModuleTabBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(globals.APP_TITLE)

        # create the main widget
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()

        # File selector
        self.open_button = QPushButton("Open Image")
        self.open_button.setFixedSize(150, 40)
        self.open_button.clicked.connect(self.selectAndBuild)
        self.main_layout.addWidget(self.open_button, alignment=Qt.AlignCenter)

        # add the layout to the central widget
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # apply global stylesheet
        file = QFile("styles/global.qss")
        if file.open(QFile.ReadOnly | QFile.Text):
            self.setStyleSheet(file.readAll().data().decode())

    def closeEvent(self, event):
        """
        Override the closeEvent from the MainWindow
        """
        event.accept()

    def selectAndBuild(self) -> None:
        """
        Opens a file dialog to select a image file and loads the main layout.
        """

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
        )

        # remove the open button from the layout
        self.main_layout.removeWidget(self.open_button)
        self.open_button.setParent(None)
        self.open_button.deleteLater()

        # ------------ Vertical Split screen ------------------
        splitter = QSplitter(Qt.Horizontal)

        # left widget
        self.image_viewer = ImageViewer()
        splitter.addWidget(self.image_viewer)

        # right widget
        self.tab_bar = ModuleTabBar(self)
        splitter.addWidget(self.tab_bar)
        self.main_layout.addWidget(splitter)

        # connect tabs from the tabbar and the image viewer
        for tab in self.tab_bar.tabs:
            self.image_viewer.image_ready.connect(tab.receive_image)
            tab.result_ready.connect(self.image_viewer.handle_result)

        self.image_viewer.load(file_path)

        # Caution: the size of the spliter should be set when the
        # layout is built otherwise it will not work correctly
        splitter.setSizes([self.width() * 2 // 3, self.width() // 3])
