from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout


class ModulTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ---------------- LAYOUT ------------------------
        main_layout = QVBoxLayout(self)

        # top row
        top_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.process)
        top_layout.addWidget(self.run_button)
        # pushes everything else to the right
        top_layout.addStretch()

        main_layout.addLayout(top_layout)
        self.setLayout(main_layout)

    def run(self):
        pass
