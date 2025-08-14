import cv2
from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, Slot
import numpy as np
from modules import NavData, SpeedData, RoadObjects, Segments, PathData


class ImageViewer(QLabel):
    # emit the image to all modules (tabs)
    image_ready = Signal(np.ndarray)

    def __init__(self, img_path=None, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.img_path = None
        self.img = None
        self.orginal_img = None

        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.advice = None
        self.speed = None
        self.objects = None
        self.segments = None
        self.paths = None

    def load(self, img_path):
        self.img_path = img_path

        if self.img_path is not None:
            self.img = cv2.imread(self.img_path)
            self.orginal_img = self.img.copy()

            if self.img is not None:
                # emit the image to all modules (tabs)
                self.image_ready.emit(self.img)
                self.redraw()

    def redraw(self):
        if self.img is None:
            return

        # get a fresh copy of the image
        self.img = self.orginal_img.copy()

        # add additional drawings to the image when avavible
        if self.speed is not None:
            self.draw_speed_data(self.speed)
        if self.advice is not None:
            self.draw_nav_data(self.advice)
        if self.objects is not None:
            self.draw_objects(self.objects)
        if self.segments is not None:
            self.draw_segments(self.segments)
        if self.paths is not None:
            self.draw_paths(self.paths)

        rgb_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        if self.img is not None:
            self.redraw()
        super().resizeEvent(event)

    @Slot(object)
    def handle_result(self, results):

        if self.img is not None:
            self.img = self.orginal_img.copy()
            # check for the type of the result
            if isinstance(results, NavData):
                self.advice = results.advice
            if isinstance(results, SpeedData):
                self.speed = results.speed
            if isinstance(results, RoadObjects):
                self.objects = results.objects
            if isinstance(results, Segments):
                self.segments = results.segments
            # to get the paths we needed segments first
            if isinstance(results, PathData) and self.segments is not None:
                self.paths = results.paths

            # redraw the new image
            self.redraw()

    def draw_nav_data(self, advice):
        text = f"{advice}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 0)
        thickness = 2

        h, w = self.img.shape[:2]

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        x = 10
        y = h - 40

        cv2.putText(
            self.img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
        )

    def draw_speed_data(self, speed):
        text = f"Speed: {speed} km/h"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 0)
        thickness = 2

        h, w = self.img.shape[:2]

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        x = 10
        y = h - 10

        cv2.putText(
            self.img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
        )

    def draw_objects(self, objects):
        for obj in objects:
            # object = {"coords": coords, "cls_name": cls_name, "conf": conf}
            # [x1, y1, x2, y2]
            x1, y1, x2, y2 = obj["coords"]

            # take the offset of 160px into account
            y1 += 160
            y2 += 160

            cv2.rectangle(self.img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_color = (0, 255, 0)
            thickness = 2

            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20  # avoid going above image

            cv2.putText(
                self.img,
                obj["cls_name"],
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    def draw_segments(self, segments):
        alpha = 0.4
        color = {
            "Driveable": (0, 255, 255),
            "Passable": (0, 255, 0),
            "Impassable": (0, 0, 255),
        }
        overlay = self.img.copy()

        # draw the driveable first
        for segment in segments:
            if segment.cls == "Driveable":
                # take the cropping into account
                pts = segment.pts.copy()
                pts[:, 0, 1] += 160

                cv2.fillPoly(overlay, [pts], color=color[segment.cls])

        for segment in segments:
            if segment.cls != "Driveable":
                # take the cropping into account
                pts = segment.pts.copy()
                pts[:, 0, 1] += 160

                cv2.fillPoly(overlay, [pts], color=color[segment.cls])

        cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0, dst=self.img)

    def draw_paths(self, paths):
        y_offset = 160
        height = self.img.shape[0]

        for p in paths:
            pts = []

            for y in range(height // 2 - y_offset, height - y_offset):
                x = p(y)
                if 0 <= x < self.img.shape[1]:
                    y_orig = y + y_offset
                    pts.append((int(x), y_orig))

            if pts:
                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    self.img, [pts_np], isClosed=False, color=(139, 0, 0), thickness=5
                )
