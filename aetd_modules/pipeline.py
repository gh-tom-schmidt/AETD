from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import cast

from cv2.typing import MatLike

from .containers import AnnotationsContainer, DirectionBox, PathsBox, RoadObjectsBox, RoadSegmentsBox, SpeedBox
from .direction import DirectionExtractor
from .paths import PathPlanner
from .road_objects import RoadObjectExtractor
from .road_segmentations import RoadSegmentsExtractor
from .speed import SpeedDataExtractor

Box = DirectionBox | SpeedBox | RoadObjectsBox | RoadSegmentsBox | PathsBox | None


class Pipeline:
    def __init__(self) -> None:
        """
        The Pipeline class orchestrates the various data extraction and processing modules.

        Methods:
            - process: Processes the input image through all extraction modules.
        """

        self.speed_data_extractor = SpeedDataExtractor()
        self.road_object_extractor = RoadObjectExtractor()
        self.road_segments_extractor = RoadSegmentsExtractor()
        self.path_planner = PathPlanner()

    def process(self, img: MatLike) -> AnnotationsContainer:
        """
        Processes the input image through all extraction modules.

        Args:
            img (MatLike): The input image to process.

        Returns:
            AnnotationsContainer: The container holding all extracted annotations.
        """

        # create a new container
        annotations_container = AnnotationsContainer(img=img)

        results: dict[str, Box] = {}
        # run all processings in parallel
        with ThreadPoolExecutor() as executor:
            # no copies of the img needed, because the modules make them
            futures: dict[Future[Box], str] = {
                executor.submit(DirectionExtractor.process, img): "direction",
                executor.submit(self.speed_data_extractor.process, img): "speed",
                executor.submit(self.road_object_extractor.process, img): "objects",
                executor.submit(self.road_segments_extractor.process, img): "segments",
            }

            # run the path planer when the segments are ready, even the rest
            # might be still processing
            for future in as_completed(fs=futures):
                key: str = futures[future]
                results[key] = future.result()
                if key == "segments":
                    annotations_container.road_segments = cast(RoadSegmentsBox | None, results["segments"])
                    if annotations_container.road_segments is not None:
                        annotations_container.paths = self.path_planner.process(
                            road_segment_box=annotations_container.road_segments,
                            width=img.shape[1],
                            height=img.shape[0],
                        )

        annotations_container.direction = cast(DirectionBox | None, results["direction"])
        annotations_container.speed = cast(SpeedBox | None, results["speed"])
        annotations_container.road_objects = cast(RoadObjectsBox | None, results["objects"])

        return annotations_container
