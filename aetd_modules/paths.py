#
# The PathExtractor class is responsible for calculating a path through a given lane.
# The PathPlanner class is responsible for planning paths given segmented lanes.
#

import math

import numpy as np
from numpy.typing import NDArray

from configs import globals

from .containers import Impassable, Passable, Path, PathsBox, RoadSegmentsBox


class PathPlanner:
    def __init__(self) -> None:
        """
        The PathPlanner class is responsible for planning paths given segmented lanes.

        Methods:
            - process: The processing pipeline for planning paths.
            - dst: Compute the overall distance of the lane to the center of the image.
            - get_distances: Calculate the distances of all impassable lanes to the center of the image.
            - strip_unreachable: Strip unreachable lanes if there are not between the closest left and right impassable lanes.
            - calculate_paths: Calculate paths between lanes and add them to the path box.
            - mirror_around_x: Mirror a polynomial function around a vertical line at x.

        """
        pass

    def process(self, road_segment_box: RoadSegmentsBox, width: int, height: int) -> PathsBox | None:
        """
        The processing pipeline for planning paths.
        """

        self.img_width: int = width
        self.img_height: int = height

        # create a new PathBox
        self.path_box: PathsBox = PathsBox()

        self.lanes: list[Impassable | Passable] = []

        # extract only the lanes
        for segment in road_segment_box:
            if isinstance(segment, Impassable) or isinstance(segment, Passable):
                self.lanes.append(segment)

        # check if there is at least two lane
        # otherwise return None for now
        # TODO: implement the "OneLane" Logic
        if len(self.lanes) < 2:
            return None

        # strip unreachable lanes
        self.strip_unreachable(distances=self.get_distances())
        # sorte the lanes by there distance to the center
        self.lanes = sorted(self.lanes, key=lambda lane: self.dst(lane=lane))
        # calculate paths between lanes
        self.calculate_paths()

        return self.path_box

    def dst(self, lane: Impassable | Passable) -> float:
        """
        Compute the overall distance of the lane to the center of the image.

        Args:
            lane (Impassable | Passable): The lane to compute the distance for.

        Returns:
            float: The distance of the lane to the center of the image.
        """

        x: NDArray[np.int32] = lane.path.approx_pts[:, 0]
        # compute mean absolute distance to center
        return float(np.mean(x - self.img_width // 2))

    def get_distances(self) -> list[float]:
        """
        Calculate the distances of all impassable lanes to the center of the image.

        Returns:
            list[float]: The distances of all lanes to the center of the image.
        """

        distances: list[float] = []
        # for all the impassable lanes
        for lane in self.lanes:
            if isinstance(lane, Impassable):
                # compute mean absolute distance to center for each lane
                distances.append(self.dst(lane=lane))

        return distances

    def strip_unreachable(self, distances: list[float]) -> None:
        """
        Strip unreachable lanes if there are not between the closest
        left and right impassable lanes.

        Args:
            distances (list[float]): The distances of the lanes to the center.

        Returns:
            None
        """

        try:
            left_impassable_lane: float | None = min([dst for dst in distances if dst <= 0])
        except ValueError:
            # if there is no left or right impassable lane
            left_impassable_lane = -math.inf

        try:
            right_impassable_lane: float | None = max([dst for dst in distances if dst > 0])
        except ValueError:
            # if there is no left or right impassable lane
            right_impassable_lane = math.inf

        # filter the lanes
        self.lanes = [
            lane for lane in self.lanes if left_impassable_lane <= self.dst(lane=lane) <= right_impassable_lane
        ]

    def calculate_paths(self) -> None:
        """
        Calculate paths between lanes and add them to the path box.
        """

        for i in range(len(self.lanes) - 1):
            # calculate the center function
            f: np.poly1d = (self.lanes[i].path.f + self.lanes[i + 1].path.f) / 2
            # create a new path and add it to the path box
            path: Path | None = PathExtractor().calculate_path_from_function(
                f=f, width=self.img_width, height=self.img_height
            )
            if path:
                self.path_box.add(path=path)

    def mirror_around_x(self, f: np.poly1d, x: float) -> np.poly1d:
        """
        Mirror a polynomial function around a vertical line at x.

        Args:
            f (np.poly1d): The polynomial function to mirror.
            x (float): The x-coordinate of the vertical line to mirror around.

        Returns:
            np.poly1d: The mirrored polynomial function.
        """

        coeffs = -f.coeffs.copy()
        coeffs[-1] += 2 * x
        return np.poly1d(c_or_r=coeffs)


class PathExtractor:
    """
    The PathExtractor class is responsible for calculating a path through a given lane.

    Methods:
        - calculate_path_from_function: Calculate a path from a polynomial function.
        - calculate_path_from_pts: Calculate a path from a set of points.
    """

    @staticmethod
    def calculate_path_from_function(f: np.poly1d, width: int, height: int) -> Path | None:
        """
        Calculate a path from a polynomial function.

        Args:
            f (np.poly1d): The polynomial function to use for path calculation.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            Path: The calculated path.
        """

        min_height: int = height // globals.HEIGHT_REDUCTION_FACTOR

        # generate approximated points for y in [0, height]
        approx_y: NDArray[np.int32] = np.linspace(
            # take the offset into account
            start=min_height,
            stop=height,
            num=height + 1,
            dtype=np.int32,
        )
        approx_x: NDArray[np.int32] = f(approx_y)

        # create mask for points within bounds
        mask = (approx_x >= 0) & (approx_x <= width) & (approx_y >= min_height) & (approx_y <= height)

        approx_x = approx_x[mask]
        approx_y = approx_y[mask]

        # create points (x, y)
        approx_pts: NDArray[np.int32] = np.stack(arrays=[approx_x, approx_y], axis=1).astype(np.int32)

        # it can be that the function we made with np.polyfit has no valid points in the given image
        # and therefore there is no valid path
        if approx_y.size > 0 and approx_x.size > 0:
            lowest_y: np.int32 = np.min(a=approx_y)
            highest_y: np.int32 = np.max(a=approx_y)
            lowest_x: np.int32 = np.min(a=approx_x)
            highest_x: np.int32 = np.max(a=approx_x)

            return Path(
                f,
                approx_pts=approx_pts,
                scope_definition=(lowest_x, highest_x),
                value_range=(lowest_y, highest_y),
            )
        else:
            return None

    @staticmethod
    def calculate_path_from_pts(pts: NDArray[np.int32], width: int, height: int) -> Path | None:
        """
        Calculate a path from a set of points.

        Args:
            pts (np.ndarray): The points to use for path calculation.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            Path: The calculated path.
        """

        x: NDArray[np.int32] = pts[:, 0]
        y: NDArray[np.int32] = pts[:, 1]

        # compute polynomial approximation x = f(y)
        coeffs: NDArray[np.float64] = np.polyfit(x=y, y=x, deg=2)

        return PathExtractor.calculate_path_from_function(f=np.poly1d(c_or_r=coeffs), width=width, height=height)
