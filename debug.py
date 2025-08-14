import sys
from PySide6.QtWidgets import QApplication
from debug import MainWindow
from configs import Config

if __name__ == "__main__":

    # load the config
    Config()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
