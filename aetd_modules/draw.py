import cv2
from cv2.typing import MatLike, NumPyArrayNumeric

from configs import globals

from .containers import AnnotationsContainer, Driveable, Impassable, Passable, Path, Sign, TrafficLight, Vehicle


class Draw:
    def __init__(self) -> None:
        """
        Draws the annotations on the image.
        """
        pass

    @staticmethod
    def draw(annotations: AnnotationsContainer) -> AnnotationsContainer:
        img: MatLike = annotations.original_img.copy()

        if annotations.direction is not None:
            img = Draw.draw_direction(img=img, advice=annotations.direction)
        if annotations.speed is not None:
            img = Draw.draw_speed(img=img, speed=annotations.speed)
        if annotations.road_objects is not None:
            img = Draw.draw_road_objects(img=img, objects=annotations.road_objects)
        if annotations.road_segments is not None:
            img = Draw.draw_road_segments(img=img, segments=annotations.road_segments)
        if annotations.paths is not None:
            img = Draw.draw_paths(img=img, paths=annotations.paths)

        annotations.annotated_img = img
        return annotations

    @staticmethod
    def draw_direction(img: MatLike, advice: int) -> MatLike:
        """
        Draw the directions as information on the image.

        Args:
            img (MatLike): The image to draw on.
            advice (int): The direction advice (-1 for left, 1 for right).

        Returns:
            MatLike: The image with the direction drawn on it.
        """

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
    def draw_speed(img: MatLike, speed: int) -> MatLike:
        """
        Draw the speed as information on the image.

        Args:
            img (MatLike): The image to draw on.
            speed (int): The speed to display.

        Returns:
            MatLike: The image with the speed drawn on it.
        """

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
    def draw_road_objects(img: MatLike, objects: list[Vehicle | Sign | TrafficLight]) -> MatLike:
        """
        Draw the road objects as bounding boxes on the image.

        Args:
            img (MatLike): The image to draw on.
            objects (list[Vehicle | Sign | TrafficLight]): The road objects to draw.

        Returns:
            MatLike: The image with the road objects drawn on it.
        """

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
    def draw_road_segments(img: MatLike, segments: list[Driveable | Passable | Impassable]) -> MatLike:
        """
        Draw the road segments on the image.

        Args:
            img (MatLike): The image to draw on.
            segments (list[Driveable | Passable | Impassable]): The road segments to draw.

        Returns:
            MatLike: The image with the road segments drawn on it.
        """

        alpha = 0.4
        colormap: dict[type, tuple[int, int, int]] = {
            Driveable: (0, 255, 255),
            Passable: (0, 255, 0),
            Impassable: (0, 0, 255),
        }

        overlay: MatLike = img.copy()

        # draw the driveable first
        for segment in segments:
            if isinstance(segment, Driveable):
                # take the cropping into account
                pts: NumPyArrayNumeric = segment.pts.copy()
                pts[:, 0, 1] += globals.ROADSEGMENT_EXTRACTION_CROP_TOP

                cv2.fillPoly(img=overlay, pts=[pts], color=colormap[type(segment)])

        for segment in segments:
            if not isinstance(segment, Driveable):
                # take the cropping into account
                pts: NumPyArrayNumeric = segment.pts.copy()
                pts[:, 0, 1] += globals.ROADSEGMENT_EXTRACTION_CROP_TOP

                cv2.fillPoly(img=overlay, pts=[pts], color=colormap[type(segment)])

        cv2.addWeighted(src1=overlay, alpha=alpha, src2=img, beta=1 - alpha, gamma=0, dst=img)

        return img

    @staticmethod
    def draw_paths(img: MatLike, paths: list[Path]) -> MatLike:
        """
        Draw the paths on the image.

        Args:
            img (MatLike): The image to draw on.
            paths (list[Path]): The paths to draw.

        Returns:
            MatLike: The image with the paths drawn on it.
        """

        for path in paths:
            cv2.polylines(img=img, pts=[path.approx_pts], isClosed=False, color=(139, 0, 0), thickness=5)

        return img
