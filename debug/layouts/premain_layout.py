from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from configs import globals


class PreloadLayout(QWidget):
    """
    The File Selector Layout (Widget) on program startup.
    """

    def __init__(self) -> None:
        super().__init__()
        self.init_ui()

    def init_ui(self) -> None:
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Dictionary to hold default file paths
        self.default_paths: dict[str, str] = {
            "image": globals.DEFAULT_IMG,
            "seg_model": globals.SEGMENTATION_MODEL_PATH,
            "det_model": globals.DETECTION_MODEL_PATH,
            "cls_model": globals.CLASSIFICATION_MODEL_PATH,
            "seg_result": globals.SEG_RESULTS,
            "det_result": globals.DET_RESULTS,
            "cls_result": globals.CLS_RESULTS,
        }

        # Dictionary to hold QLabel references for file paths
        self.file_labels: dict[str, QLabel] = {}

        # Dictionary to hold QCheckBox references for results
        self.file_checkboxes: dict[str, QCheckBox] = {}

        # Create buttons and labels
        self.add_file_selector("Select Image", "image", "Images (*.jpg *.jpeg)")
        self.add_file_selector("Select Segmentation Model", "seg_model", "PyTorch Models (*.pt)")
        self.add_file_selector("Select Detection Model", "det_model", "PyTorch Models (*.pt)")
        self.add_file_selector("Select Classification Model", "cls_model", "PyTorch Models (*.pt)")
        self.add_file_selector("Select Segmentation Result", "seg_result", "Pickle Files (*.pkl)", with_checkbox=True)
        self.add_file_selector("Select Detection Result", "det_result", "Pickle Files (*.pkl)", with_checkbox=True)
        self.add_file_selector("Select Classification Result", "cls_result", "Pickle Files (*.pkl)", with_checkbox=True)

    def add_file_selector(self, button_text: str, key: str, file_filter: str, with_checkbox: bool = False) -> None:
        h_layout = QHBoxLayout()
        # add margin around the layout
        h_layout.setContentsMargins(20, 20, 20, 20)
        h_layout.setSpacing(12)

        btn = QPushButton(button_text)
        btn.setFixedWidth(220)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # make label text bigger
        font = QFont()
        font.setPointSize(10)

        label = QLabel(self.default_paths[key])
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label.setFont(font)

        btn.clicked.connect(lambda k=key, f=file_filter: self.select_file(k, f))

        h_layout.addWidget(btn)

        if with_checkbox:
            checkbox = QCheckBox()
            # default: apply results unless unchecked
            checkbox.setChecked(True)
            h_layout.addWidget(checkbox)
            self.file_checkboxes[key] = checkbox

        h_layout.addWidget(label)
        self.main_layout.addLayout(h_layout)

        self.file_labels[key] = label

    def select_file(self, key: str, file_filter: str) -> None:
        file_path: str = QFileDialog.getOpenFileName(self, f"Select {key}", "", file_filter)[0]
        self.file_labels[key].setText(file_path)

    def setGlobals(self) -> None:
        """
        Write selected file paths back into global constants.
        """

        globals.DEFAULT_IMG = self.file_labels["image"].text()
        globals.SEGMENTATION_MODEL_PATH = self.file_labels["seg_model"].text()
        globals.DETECTION_MODEL_PATH = self.file_labels["det_model"].text()
        globals.CLASSIFICATION_MODEL_PATH = self.file_labels["cls_model"].text()

        # Only apply results if their checkbox is checked else set to none to override the global default
        if self.file_checkboxes["seg_result"].isChecked():
            globals.SEG_RESULTS = self.file_labels["seg_result"].text()
        else:
            globals.SEG_RESULTS = None

        if self.file_checkboxes["det_result"].isChecked():
            globals.DET_RESULTS = self.file_labels["det_result"].text()
        else:
            globals.DET_RESULTS = None

        if self.file_checkboxes["cls_result"].isChecked():
            globals.CLS_RESULTS = self.file_labels["cls_result"].text()
        else:
            globals.CLS_RESULTS = None
