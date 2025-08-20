from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from configs import globals


class InfoTable(QWidget):
    """
    A simple info table to show some global variables.
    """

    def __init__(self) -> None:
        """
        A simple info table to show some global variables.
        """

        super().__init__()

        data: dict[str, str] = {
            "image": globals.DEFAULT_IMG,
            "seg_model": globals.SEGMENTATION_MODEL_PATH,
            "det_model": globals.DETECTION_MODEL_PATH,
            "cls_model": globals.CLASSIFICATION_MODEL_PATH,
            "seg_result": globals.SEG_RESULTS,
            "det_result": globals.DET_RESULTS,
            "cls_result": globals.CLS_RESULTS,
        }

        # create table
        table = QTableWidget()
        table.setRowCount(len(data))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Key", "Value"])

        for row, (key, value) in enumerate(data.items()):
            table.setItem(row, 0, QTableWidgetItem(key))
            table.setItem(row, 1, QTableWidgetItem(str(value)))

        # make first column resize to contents, second column take the rest
        header: QHeaderView = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        # layout
        layout = QVBoxLayout()
        layout.addWidget(table)
        self.setLayout(layout)
