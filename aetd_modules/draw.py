import cv2
from numpy import uint8
from numpy.typing import NDArray

from configs import globals

from .containers import AnnotationsContainer, Driveable, Impassable, Passable, Path, Sign, TrafficLight, Vehicle
from .types import Img


class Draw:
    def __init__(self) -> None:
        """
        Draws the annotations on the image.
        """
        pass

    @staticmethod
    def draw(annotations: AnnotationsContainer) -> AnnotationsContainer:
        img: Img = annotations.original_img.copy()

        if annotations.direction:
            Draw.draw_direction(img=img, advice=annotations.direction)
        if annotations.speed:
            Draw.draw_speed(img=img, speed=annotations.speed)
        if annotations.road_objects:
            Draw.draw_road_objects(img=img, objects=annotations.road_objects)
        if annotations.road_segments:
            Draw.draw_road_segments(img=img, segments=annotations.road_segments)
        if annotations.paths:
            Draw.draw_paths(img=img, paths=annotations.paths)
        return annotations

    @staticmethod
    def draw_direction(img: Img, advice: int) -> Img:
        if advice == -1:
            text = "Turn Left"
        elif advice == 1:
            text = "Turn Right"
        elif advice == 0:
            text = "Go Straight"
        else:
            raise ValueError("Invalid direction advice")

        font: int = cv2.FONT_HERSHEY_SIMPLEX
        font_scale: float = 1.0
        color: tuple[int, int, int] = (255, 255, 0)
        thickness: int = 2

        x: int = 10
        y: int = img.shape[0] - 40

        cv2.putText(
            img=img,
            text=text,
            org=(x, y),
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        return img

    @staticmethod
    def draw_speed(img: Img, speed: int) -> Img:
        text: str = f"Speed: {speed} km/h"

        font: int = cv2.FONT_HERSHEY_SIMPLEX
        font_scale: float = 1.0
        color: tuple[int, int, int] = (255, 255, 0)
        thickness: int = 2

        x: int = 10
        y: int = img.shape[0] - 80

        cv2.putText(
            img=img,
            text=text,
            org=(x, y),
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        return img

    @staticmethod
    def draw_road_objects(img: Img, objects: list[Vehicle | Sign | TrafficLight]) -> Img:
        for obj in objects:
            # [x1, y1, x2, y2]
            x1, y1, x2, y2 = obj.coords

            # take the offset of 160px into account
            y1 += globals.ROADOBJECT_EXTRACTION_CROP_TOP
            y2 += globals.ROADOBJECT_EXTRACTION_CROP_TOP

            cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

            font: int = cv2.FONT_HERSHEY_SIMPLEX
            font_scale: float = 0.6
            text_color: tuple[int, int, int] = (0, 255, 0)
            thickness: int = 2

            text_x: int = x1
            text_y: int = y1 - 10 if y1 - 10 > 10 else y1 + 20  # avoid going above image

            cv2.putText(
                img=img,
                text=str(object=obj.cls),
                org=(text_x, text_y),
                fontFace=font,
                fontScale=font_scale,
                color=text_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        return img

    @staticmethod
    def draw_road_segments(img: Img, segments: list[Driveable | Passable | Impassable]) -> Img:
        alpha = 0.4
        colormap: dict[type, tuple[int, int, int]] = {
            Driveable: (0, 255, 255),
            Passable: (0, 255, 0),
            Impassable: (0, 0, 255),
        }

        overlay: Img = img.copy()

        # draw the driveable first
        for segment in segments:
            if isinstance(segment, Driveable):
                # take the cropping into account
                pts: NDArray[uint8] = segment.pts.copy()
                pts[:, 0, 1] += globals.ROADSEGMENT_EXTRACTION_CROP_TOP

                cv2.fillPoly(img=overlay, pts=[pts], color=colormap[type(segment)])

        for segment in segments:
            if not isinstance(segment, Driveable):
                # take the cropping into account
                pts: NDArray[uint8] = segment.pts.copy()
                pts[:, 0, 1] += globals.ROADSEGMENT_EXTRACTION_CROP_TOP

                cv2.fillPoly(img=overlay, pts=[pts], color=colormap[type(segment)])

        cv2.addWeighted(src1=overlay, alpha=alpha, src2=img, beta=1 - alpha, gamma=0, dst=img)

        return img

    @staticmethod
    def draw_paths(img: Img, paths: list[Path]) -> Img:
        for path in paths:
            cv2.polylines(img=img, pts=[path.approx_pts], isClosed=False, color=(139, 0, 0), thickness=5)

        return img
