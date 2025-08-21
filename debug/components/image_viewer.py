import os

import cv2
from cv2.typing import MatLike, NumPyArrayNumeric
from PySide6.QtCore import Qt, Signal, Slot  # pyright: ignore[reportUnknownVariableType]
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from aetd_modules import AnnotationsContainer, Draw
from configs import globals


class ImageViewer(QLabel):
    # emit the image to all modules (tabs)
    image_ready = Signal(AnnotationsContainer)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        # load the given global image path
        img: cv2.Mat | NumPyArrayNumeric | None = cv2.imread(filename=globals.DEFAULT_IMG, flags=cv2.IMREAD_COLOR)
        if img is not None:
            basename = os.path.basename(globals.DEFAULT_IMG)
            self.annotations_container: AnnotationsContainer = AnnotationsContainer(img=img, img_name=basename)
        else:
            raise ValueError(f"Image at {globals.DEFAULT_IMG} could not be loaded.")

    def emit_annotation_container(self) -> None:
        self.image_ready.emit(self.annotations_container)

    def redraw(self) -> None:
        Draw.draw(annotations=self.annotations_container)

        rgb_image: MatLike = cv2.cvtColor(src=self.annotations_container.annotated_img, code=cv2.COLOR_BGR2RGB)

        h: int = rgb_image.shape[0]
        w: int = rgb_image.shape[1]
        ch: int = rgb_image.shape[2]
        bytes_per_line: int = ch * w

        q_img = QImage(
            rgb_image.tobytes(),
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        self.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.redraw()
        super().resizeEvent(event)

    @Slot(AnnotationsContainer)
    def update_container(self, annotations_container: AnnotationsContainer) -> None:
        self.annotations_container = annotations_container
        self.redraw()
