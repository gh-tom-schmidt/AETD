#
# This file holds the data containers storing various values from the image processing pipeline
#

from modules import Driveable, Impassable, Passable, Path, Sign, TrafficLight, Vehicle
from tools import DirectionOutOfBounds, Img, SpeedOutOfBounds


class AnnotationsContainer:
    def __init__(self, img: Img) -> None:
        """
        The AnnotationsContainer class holds various data attributes related to the image processing pipeline.

        Args:
            img (Img): The original image.

        Attributes:
            original_img (Img | None): The original image.
            annotated_img (Img | None): The annotated image.
            speed (SpeedBox | None): The speed value.
            direction (DirectionBox | None): The direction value.
            road_objects (RoadObjectsBox | None): The road objects.
            road_segments (RoadSegmentsBox | None): The road segments.
            paths (PathsBox | None): The paths.
        """

        self.original_img: Img = img
        # on initialization the original and annotated images are the same
        self.annotated_img: Img = img.copy()

        self.speed: SpeedBox | None = None
        self.direction: DirectionBox | None = None
        self.road_objects: RoadObjectsBox | None = None
        self.road_segments: RoadSegmentsBox | None = None
        self.paths: PathsBox | None = None


class SpeedBox(int):
    """
    This class holds the speed value retrieved from the image and
    ensures it is within the valid range (0-100).

    Methods:
        - __new__: Creates a new SpeedBox instance given the speed.
    """

    def __new__(cls, speed: int) -> None:
        """
        Creates a new SpeedBox instance.

        Args:
            speed (int | None): The speed value to be wrapped.
        """

        if 0 <= speed <= 100:
            super().__new__(cls, speed)
        else:
            raise SpeedOutOfBounds(speed)


class DirectionBox(int):
    """
    This class holds the direction value retrieved from the image and
    ensures that it has one of the three correct direction values (-1, 0, 1).

    Methods:
        - __new__: Creates a new DirectionBox instance given the direction.
    """

    def __new__(cls, direction: int) -> None:
        """
        Creates a new DirectionBox instance.

        Args:
            direction (int | None): The direction value to be wrapped.
        """

        if direction in [-1, 0, 1]:
            super().__new__(cls, direction)
        else:
            raise DirectionOutOfBounds(direction=direction)


class RoadObjectsBox(list[Vehicle | Sign | TrafficLight]):
    """
    This class holds a list of road objects.

    Methods:
        - add: Adds a road object to the list.
    """

    def add(self, road_object: Vehicle | Sign | TrafficLight) -> None:
        """
        Add a road object to the list.

        Args:
            road_object (Vehicle | Sign | TrafficLight): The road object to be added.
        """

        self.append(road_object)


class RoadSegmentsBox(list[Driveable | Passable | Impassable]):
    """
    This class holds a list of road segments.

    Methods:
        - add: Adds a road segment to the list.
    """

    def add(self, road_segment: Driveable | Passable | Impassable) -> None:
        """
        Add a road segment to the list.

        Args:
            road_segment (Driveable | Passable | Impassable): The road segment to be added.
        """

        self.append(road_segment)


class PathsBox(list[Path]):
    """
    This class holds a list of paths.

    Methods:
        - add: Adds a path to the list.
    """

    def add(self, path: Path) -> None:
        """
        Add a path to the list.

        Args:
            path (Path): The path to be added.
        """

        self.append(path)
