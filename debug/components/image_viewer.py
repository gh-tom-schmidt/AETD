import cv2
from cv2.typing import NumPyArrayNumeric
from PySide6.QtCore import Qt, Signal, Slot  # pyright: ignore[reportUnknownVariableType]
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from aetd_modules import AnnotationsContainer, Draw, Img, ImgT


class ImageViewer(QLabel):
    # emit the image to all modules (tabs)
    image_ready = Signal(AnnotationsContainer)

    def __init__(self, img_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_path = None

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        # load the given image
        img: cv2.Mat | NumPyArrayNumeric | None = cv2.imread(filename=img_path, flags=cv2.IMREAD_COLOR)
        if img is not None:
            self.annotations_container: AnnotationsContainer = AnnotationsContainer(img=ImgT(img=img))
        else:
            raise ValueError(f"Image at {img_path} could not be loaded.")

    def emit_annotation_container(self) -> None:
        self.image_ready.emit(self.annotations_container)

    def redraw(self) -> None:
        Draw.draw(annotations=self.annotations_container)

        rgb_image: Img = ImgT(img=cv2.cvtColor(src=self.annotations_container.annotated_img, code=cv2.COLOR_BGR2RGB))

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
