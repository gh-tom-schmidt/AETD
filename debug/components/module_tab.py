from abc import abstractmethod

from cv2.typing import MatLike
from PySide6.QtCore import Signal, Slot  # pyright: ignore[reportUnknownVariableType]
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aetd_modules import AnnotationsContainer


class ModulTab(QWidget):
    # emit the results
    result_ready = Signal(AnnotationsContainer)

    def __init__(self, parent: QWidget | None = None, table_data: list[tuple[str, str]] | None = None) -> None:
        super().__init__(parent)
        self.img: MatLike | None = None

        # ---------------------- LAYOUT ------------------------
        main_layout = QVBoxLayout(self)

        # ---------------------- Table -------------------------

        # if there is table data, create a table
        if table_data is not None:
            # set the basic tabel
            self.info_table = QTableWidget(len(table_data), 2)
            self.info_table.setHorizontalHeaderLabels(["Name", "Value"])
            self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.info_table.verticalHeader().setVisible(False)

            # fill the table from the list
            for row_index, (name, value) in enumerate(table_data):
                self.info_table.setItem(row_index, 0, QTableWidgetItem(name))
                self.info_table.setItem(row_index, 1, QTableWidgetItem(value))

            # hook the table into the layout
            main_layout.addWidget(self.info_table)

        # ------------------- Run Button ---------------------
        self.run_button = QPushButton(text="Run")
        self.run_button.setFixedHeight(40)  # give it some height
        self.run_button.clicked.connect(
            lambda checked=False: self.process(annotations_container=self.annotations_container)
        )

        # make button full width
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.run_button)

        main_layout.addStretch()  # keeps button at bottom
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    # get the annotation container from viewer
    @Slot(AnnotationsContainer)
    def receive_annotation_container(self, annotations_container: AnnotationsContainer) -> None:
        self.annotations_container: AnnotationsContainer = annotations_container

    @abstractmethod
    def process(self, annotations_container: AnnotationsContainer) -> None:
        """
        Default implementation (can be overridden).
        """
        pass
