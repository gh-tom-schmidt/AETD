from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING

import cv2
from cv2.typing import MatLike, NumPyArrayNumeric
from PySide6.QtCore import Qt, QTimerEvent
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget

from aetd_modules import AnnotationsContainer, Draw
from configs import globals

if TYPE_CHECKING:
    from debug.components.tab_bars import ModuleTabBar


class View(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self.annotations_container: AnnotationsContainer | None = None

    def container(self, annotations_container: AnnotationsContainer) -> None:
        self.annotations_container = annotations_container
        self.redraw()

    def redraw(self) -> None:
        if self.annotations_container is not None:
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


class ImageViewerTab(QWidget):
    def __init__(self, module_tab_bar: "ModuleTabBar", parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.module_tab_bar = module_tab_bar
        self.view = View(self)

        # Create reload button
        self.reload_btn = QPushButton("⟳")
        self.reload_btn.setFixedSize(30, 30)
        self.reload_btn.clicked.connect(self.reload_image)

        # Top bar with reload button on the right
        top_bar = QHBoxLayout()
        top_bar.addStretch()  # pushes button to the right
        top_bar.addWidget(self.reload_btn)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.view)
        self.setLayout(layout)

        # load the given global image path if the image path is not none
        self.reload_image()

    def reload_image(self) -> None:
        if globals.DEFAULT_IMG != "":
            img: cv2.Mat | NumPyArrayNumeric | None = cv2.imread(filename=globals.DEFAULT_IMG, flags=cv2.IMREAD_COLOR)
            if img is not None:
                basename: str = os.path.basename(globals.DEFAULT_IMG)
                self.view.container(
                    annotations_container=self.module_tab_bar.process(
                        annotations_container=AnnotationsContainer(img=img, img_name=basename)
                    )
                )
            else:
                self.view.setText("Failed to load image.")
        else:
            self.view.setText("No image loaded.")


class VideoViewerTab(QWidget):
    def __init__(self, module_tab_bar: "ModuleTabBar", parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.view = View(self)
        self.module_tab_bar: "ModuleTabBar" = module_tab_bar

        # ----------------- LOAD VIDEO ----------------------
        # load the given global video path
        if globals.DEFAULT_VIDEO != "":
            self.cap = cv2.VideoCapture(globals.DEFAULT_VIDEO)
            if self.cap.isOpened():
                self.next_frame()
            else:
                raise ValueError("Failed to load video.")
        else:
            self.view.setText("No video loaded.")

        # ------------------- LAYOUT ------------------------
        layout = QVBoxLayout()
        layout.addWidget(self.view)

        # ------------------- SLIDER ------------------------
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        layout.addWidget(self.slider)

        # ---------------- BUTTON LAYOUT --------------------
        controlls = QHBoxLayout()
        controlls.setSpacing(10)

        # skip back buttons
        for skip in [-60, -30, -10, -1]:
            btn = QPushButton(f"{skip}")
            btn.clicked.connect(partial(self.skip, skip))
            controlls.addWidget(btn)

        # play button
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play)
        controlls.addWidget(self.play_btn)

        # skip forward buttons
        for skip in [1, 10, 30, 60]:
            btn = QPushButton(f"+{skip}")
            btn.clicked.connect(partial(self.skip, skip))
            controlls.addWidget(btn)

        # center the row
        controlls.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(controlls)
        self.setLayout(layout)

    def next_frame(self) -> None:
        ret, self.frame = self.cap.read()
        if ret:
            self.view.container(
                annotations_container=self.module_tab_bar.process(
                    annotations_container=AnnotationsContainer(img=self.frame, img_name="frame")
                )
            )

    def play(self) -> None:
        if self.play_btn.text() == "Play":
            self.play_btn.setText("Pause")
            self.timer: int = self.startTimer(30)  # 30 ms for ~33 FPS
        else:
            self.play_btn.setText("Play")
            self.killTimer(self.timer)

    def skip(self, delta: int) -> None:
        self.cap.set(propId=cv2.CAP_PROP_POS_FRAMES, value=self.slider.value() + delta)
        self.next_frame()

    def timerEvent(self, event: QTimerEvent) -> None:
        event.accept()
        self.next_frame()
