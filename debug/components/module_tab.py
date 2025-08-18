from PySide6.QtCore import Signal, Slot  # pyright: ignore[reportUnknownVariableType]
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from modules import AnnotationsContainer, DirectionBox, PathPlanner, RoadObjectsBox, RoadSegmentsBox, SpeedBox
from tools import Box, Extractors, Img


class ModulTab(QWidget):
    # emit the results
    result_ready = Signal(AnnotationsContainer)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.img: Img | None = None
        self.modul: Extractors | None = None

        # ---------------- LAYOUT ------------------------
        main_layout = QVBoxLayout(parent=self)

        # top row
        top_layout = QHBoxLayout()
        self.run_button = QPushButton(text="Run")
        self.run_button.clicked.connect(slot=self.process)
        top_layout.addWidget(arg__1=self.run_button)
        # pushes everything else to the right
        top_layout.addStretch()

        main_layout.addLayout(layout=top_layout)
        self.setLayout(arg__1=main_layout)

    @Slot(t1=AnnotationsContainer)
    def receive_annotation_container(self, annotations_container: AnnotationsContainer) -> None:
        self.annotations_container: AnnotationsContainer = annotations_container

    def process(self) -> None:
        if self.modul is not None:
            if isinstance(self.modul, PathPlanner):
                # path planner needs the segments
                if self.annotations_container.road_segments is not None:
                    self.annotations_container.paths = self.modul.process(
                        road_segment_box=self.annotations_container.road_segments,
                        width=self.annotations_container.original_img.shape[1],
                        height=self.annotations_container.original_img.shape[0],
                    )
            else:
                result: Box | AnnotationsContainer = self.modul.process(self.annotations_container.original_img)

                if isinstance(result, DirectionBox):
                    self.annotations_container.direction = result
                elif isinstance(result, SpeedBox):
                    self.annotations_container.speed = result
                elif isinstance(result, RoadObjectsBox):
                    self.annotations_container.road_objects = result
                elif isinstance(result, RoadSegmentsBox):
                    self.annotations_container.road_segments = result
                elif isinstance(result, AnnotationsContainer):
                    self.annotations_container = result

        self.result_ready.emit(self.annotations_container)
