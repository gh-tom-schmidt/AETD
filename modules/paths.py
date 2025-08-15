#
# The PathExtractor class is responsible for calculating a path through a given lane.
# The PathPlanner class is responsible for planning paths given segmented lanes.
#

import numpy as np
import math

from .data_containers import PathsBox, RoadSegmentsBox
from .road_segmentations import Impassable, Passable


class PathPlanner:
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

    def __init__(self) -> None:
        """
        Initialize the PathPlanner.
        """

        self.path_box = None
        self.lanes = None

    def process(self, road_segment_box: RoadSegmentsBox) -> PathsBox:
        """
        The processing pipeline for planning paths.
        """

        # create a new PathBox
        self.path_box = PathsBox()

        self.lanes = []

        # extract only the lanes
        for segment in road_segment_box:
            if isinstance(segment, Impassable) or isinstance(segment, Passable):
                self.lanes.append[segment]

        # check if there is at least one lane
        # otherwise return None for now
        # TODO: implement the "OneLane" Logic
        if len(self.lanes) >= 1:
            return None

        # strip unreachable lanes
        self.strip_unreachable(self.get_distances())
        # sorte the lanes by there distance to the center
        self.lanes = sorted(self.lanes, key=lambda l: self.dst(l))

        return self.path_box

    def dst(self, lane: Impassable | Passable) -> float:
        """
        Compute the overall distance of the lane to the center of the image.

        Args:
            lane (Impassable | Passable): The lane to compute the distance for.

        Returns:
            float: The distance of the lane to the center of the image.
        """

        x = lane.path.approx_pts[:, 0]
        # compute mean absolute distance to center
        return np.mean(x - self.img_width // 2)

    def get_distances(self) -> list[float]:
        """
        Calculate the distances of all impassable lanes to the center of the image.

        Returns:
            list[float]: The distances of all lanes to the center of the image.
        """

        distances = []
        # for all the impassable lanes
        for lane in self.lanes:
            if isinstance(lane, Impassable):
                # compute mean absolute distance to center for each lane
                distances.append(self.dst(lane))

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

        left_impassable_lane = min([dst for dst in distances if dst <= 0])
        right_impassable_lane = max([dst for dst in distances if dst > 0])

        # if there is no left or right impassable lane
        if left_impassable_lane is None:
            left_impassable_lane = -math.inf
        if right_impassable_lane is None:
            right_impassable_lane = math.inf

        # filter the lanes
        self.lanes = [
            l
            for l in self.lanes
            if left_impassable_lane <= self.dst(l) <= right_impassable_lane
        ]

    def calculate_paths(self) -> None:
        """
        Calculate paths between lanes and add them to the path box.
        """

        for i in range(len(self.lanes) - 1):
            # calculate a center function between to lanes
            f = (self.lanes[i].path.f + self.lanes[i + 1].path.f) / 2
            # create a new path and add it to the path box
            self.path_box.add(PathExtractor().calculate_path_from_function(f))

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
        return np.poly1d(coeffs)


class Path:
    """
    This class holds a polynomial function and its approximated points with bounds.
    """

    def __init__(
        self,
        f: np.poly1d,
        approx_pts: np.ndarray,
        scope_definition: tuple,
        value_range: tuple,
    ) -> None:
        self.f = f
        self.approx_pts = approx_pts
        self.scope_definition = scope_definition
        self.value_range = value_range


class PathExtractor:
    """
    The PathExtractor class is responsible for calculating a path through a given lane.

    Methods:
        - calculate_path_from_function: Calculate a path from a polynomial function.
        - calculate_path_from_pts: Calculate a path from a set of points.
    """

    def calculate_path_from_function(
        self, f: np.poly1d, width: int, height: int
    ) -> Path:
        """
        Calculate a path from a polynomial function.

        Args:
            f (np.poly1d): The polynomial function to use for path calculation.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            Path: The calculated path.
        """

        # generate approximated points for y in [0, height]
        approx_y = np.linspace(0, height, num=height + 1)
        approx_x = f(approx_y)

        # create mask for points within bounds
        mask = (
            (approx_x >= 0)
            & (approx_x <= width)
            & (approx_y >= 0)
            & (approx_y <= height)
        )

        approx_x = approx_x[mask]
        approx_y = approx_y[mask]

        # create points (x, y)
        approx_pts = np.stack([approx_x, approx_y], axis=1)

        lowest_y = np.min(approx_y) if approx_y.size > 0 else None
        highest_y = np.max(approx_y) if approx_y.size > 0 else None
        lowest_x = np.min(approx_x) if approx_x.size > 0 else None
        highest_x = np.max(approx_x) if approx_x.size > 0 else None

        return Path(f, approx_pts, (lowest_x, highest_x), (lowest_y, highest_y))

    def calculate_path_from_pts(self, pts: np.ndarray, width: int, height: int) -> Path:
        """
        Calculate a path from a set of points.

        Args:
            pts (np.ndarray): The points to use for path calculation.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            Path: The calculated path.
        """

        x = pts[:, 0]
        y = pts[:, 1]

        # compute polynomial approximation x = f(y)
        coeffs = np.polyfit(y, x, deg=2)

        return self.calculate_path_from_function(np.poly1d(coeffs), width, height)
