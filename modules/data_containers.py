#
# This file holds the data containers storing various values from the image processing pipeline
#

from exceptions import DirectionOutOfBounds, SpeedOutOfBounds
from typing import List


class DataContainer:
    """
    The DataContainer class holds various data attributes related to the image processing pipeline.

    Methods:
        - __init__: Initializes the data container with default value: None.
    """

    def __init__(self) -> None:
        """
        Initializes the data container with default value: None.
        """

        self.original_img = None
        self.annotated_img = None

        self.speed = None
        self.direction = None
        self.road_objects = None
        self.road_segments = None
        self.paths = None


class SpeedBox(int):
    """
    This class holds the speed value retrieved from the image and
    ensures it is within the valid range (0-100).

    Methods:
        - __new__: Creates a new SpeedBox instance given the speed.
    """

    def __new__(cls, speed: int | None) -> None:
        """
        Creates a new SpeedBox instance.

        Args:
            speed (int | None): The speed value to be wrapped.
        """

        if 0 <= speed <= 100 or speed is None:
            return super().__new__(cls, speed)
        else:
            raise SpeedOutOfBounds(speed)


class DirectionBox(int):
    """
    This class holds the direction value retrieved from the image and
    ensures that it has one of the three correct direction values (-1, 0, 1).

    Methods:
        - __new__: Creates a new DirectionBox instance given the direction.
    """

    def __new__(cls, direction: int | None) -> None:
        """
        Creates a new DirectionBox instance.

        Args:
            direction (int | None): The direction value to be wrapped.
        """

        if direction in [-1, 0, 1] or direction is None:
            return super().__new__(cls, direction)
        else:
            raise DirectionOutOfBounds(direction)


class RoadObjectsBox(List[RoadObject]):
    """
    This class holds a list of road objects.

    Methods:
        - add: Adds a road object to the list.
    """

    def add(self, road_object: RoadObject) -> None:
        """
        Add a road object to the list.

        Args:
            road_object (RoadObject): The road object to be added.
        """

        self.append(road_object)


class RoadSegmentsBox(List[RoadSegmetns]):
    """
    This class holds a list of road segments.

    Methods:
        - add: Adds a road segment to the list.
    """

    def add(self, road_segment: RoadSegment) -> None:
        """
        Add a road segment to the list.

        Args:
            road_segment (RoadSegment): The road segment to be added.
        """

        self.append(road_segment)


class PathsBox(List[Path]):
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
