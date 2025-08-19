#
# This file holds the data containers storing various values from the image processing pipeline
#

import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray


class AnnotationsContainer:
    def __init__(self, img: MatLike) -> None:
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

        self.original_img: MatLike = img
        # on initialization the original and annotated images are the same
        self.annotated_img: MatLike = img.copy()

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

    def __new__(cls, speed: int) -> "SpeedBox":
        """
        Creates a new SpeedBox instance.

        Args:
            speed (int | None): The speed value to be wrapped.
        """

        if 0 <= speed <= 100:
            return super().__new__(cls, speed)
        else:
            raise ValueError(f"Invalid speed value: {speed}")


class DirectionBox(int):
    """
    This class holds the direction value retrieved from the image and
    ensures that it has one of the three correct direction values (-1, 0, 1).

    Methods:
        - __new__: Creates a new DirectionBox instance given the direction.
    """

    def __new__(cls, direction: int) -> "DirectionBox":
        """
        Creates a new DirectionBox instance.

        Args:
            direction (int | None): The direction value to be wrapped.
        """

        if direction in [-1, 0, 1]:
            return super().__new__(cls, direction)
        else:
            raise ValueError(f"Invalid direction value: {direction}")


class Path:
    """
    This class holds a polynomial function and its approximated points with bounds.
    """

    def __init__(
        self,
        f: np.poly1d,
        approx_pts: NDArray[np.int32],
        scope_definition: tuple[np.int32, np.int32],
        value_range: tuple[np.int32, np.int32],
    ) -> None:
        self.f: np.poly1d = f
        self.approx_pts: NDArray[np.int32] = approx_pts
        self.scope_definition: tuple[np.int32, np.int32] = scope_definition
        self.value_range: tuple[np.int32, np.int32] = value_range


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


class Vehicle:
    """
    This class holds the information for detected vehicles.

    Args:
        coords (tuple[int, int, int, int]): The coordinates of the vehicle.
        cls (int): The class ID of the vehicle.
    """

    def __init__(self, coords: tuple[int, int, int, int], cls: int) -> None:
        """
        This class holds the information for detected vehicles.

        Args:
            coords (tuple[int, int, int, int]): The coordinates of the vehicle.
            cls (int): The class ID of the vehicle.
        """

        self.coords: tuple[int, int, int, int] = coords
        self.cls: int = cls


class Sign:
    """
    This class holds the information for detected signs.

    Args:
        coords (tuple[int, int, int, int]): The coordinates of the sign.
        cls (int): The class ID of the sign.
    """

    def __init__(self, coords: tuple[int, int, int, int], cls: int) -> None:
        """
        This class holds the information for detected signs.

        Args:
            coords (tuple[int, int, int, int]): The coordinates of the sign.
            cls (int): The class ID of the sign.
        """

        self.coords: tuple[int, int, int, int] = coords
        self.cls: int = cls


class TrafficLight:
    """
    This class holds the information for detected traffic lights.

    Args:
        coords (tuple[int, int, int, int]): The coordinates of the traffic light.
        cls (int): The class ID of the traffic light.
    """

    def __init__(self, coords: tuple[int, int, int, int], cls: int) -> None:
        """
        This class holds the information for detected traffic lights.

        Args:
            coords (tuple[int, int, int, int]): The coordinates of the traffic light.
            cls (int): The class ID of the traffic light.
        """

        self.coords: tuple[int, int, int, int] = coords
        self.cls: int = cls


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


class Driveable:
    """
    This class holds the information for a driveable area.

    Args:
        pts (NDArray[np.int32]): The points of the polygon.
        path (Path): The path associated with the polygon.
    """

    def __init__(self, pts: NDArray[np.int32], path: Path) -> None:
        self.pts: NDArray[np.int32] = pts
        self.path: Path = path


class Impassable:
    """
    This class holds the information for a impassable lane.

    Args:
        pts (NDArray[np.int32]): The points of the polygon.
        path (Path): The path associated with the polygon.
    """

    def __init__(self, pts: NDArray[np.int32], path: Path) -> None:
        self.pts: NDArray[np.int32] = pts
        self.path: Path = path


class Passable:
    """
    This class holds the information for a passable lane.

    Args:
        pts (NDArray[np.int32]): The points of the polygon.
        path (Path): The path associated with the polygon.
    """

    def __init__(self, pts: NDArray[np.int32], path: Path) -> None:
        self.pts: NDArray[np.int32] = pts
        self.path: Path = path


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
