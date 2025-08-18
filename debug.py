import sys

from PySide6.QtWidgets import QApplication

from configs import Config
from debug import MainWindow

if __name__ == "__main__":
    # load the config
    Config(conf_file="configs/debug_default.conf")

    app = QApplication(arg__1=sys.argv)
    window: MainWindow = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
