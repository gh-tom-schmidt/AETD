from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import cast

from cv2.typing import MatLike
from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]

from .containers import AnnotationsContainer, DirectionBox, PathsBox, RoadObjectsBox, RoadSegmentsBox, SpeedBox
from .direction import DirectionExtractor
from .paths import PathPlanner
from .road_object_classification import RoadObjectClassificationRefiner
from .road_objects_detection import RoadObjectDetectionExtractor
from .road_segmentations import RoadSegmentsExtractor
from .speed import SpeedDataExtractor

Box = DirectionBox | SpeedBox | RoadObjectsBox | RoadSegmentsBox | PathsBox | None


class Pipeline:
    def __init__(
        self,
        only_cls_results: str | bool = False,
        only_seg_results: str | bool = False,
        only_det_results: str | bool = False,
    ) -> None:
        """
        The Pipeline class orchestrates the various data extraction and processing modules.

        Methods:
            - process: Processes the input image through all extraction modules.
        """

        self.speed_data_extractor = SpeedDataExtractor()
        self.road_object_detection_extractor = RoadObjectDetectionExtractor(only_detec_results=only_det_results)
        self.road_classification_refiner = RoadObjectClassificationRefiner(only_cls_results=only_cls_results)
        self.road_segments_extractor = RoadSegmentsExtractor(only_results=only_seg_results)
        self.path_planner = PathPlanner()

    def process(
        self,
        img: MatLike,
        img_name: str,
        detect_result: Results | None = None,
        cls_result: Results | None = None,
        seg_result: Results | None = None,
    ) -> AnnotationsContainer:
        """
        Processes the input image through all extraction modules.

        Args:
            img (MatLike): The input image to process.

        Returns:
            AnnotationsContainer: The container holding all extracted annotations.
        """

        # create a new container
        annotations_container = AnnotationsContainer(img=img, img_name=img_name)

        results: dict[str, Box] = {}
        # run all processings in parallel
        with ThreadPoolExecutor() as executor:
            # no copies of the img needed, because the modules make them
            futures: dict[Future[Box], str] = {
                executor.submit(DirectionExtractor.process, img): "direction",
                executor.submit(self.speed_data_extractor.process, img): "speed",
                executor.submit(self.road_object_detection_extractor.process, img, detect_result): "objects",
                executor.submit(self.road_segments_extractor.process, img, seg_result): "segments",
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
                if key == "objects":
                    annotations_container.road_objects = cast(RoadObjectsBox | None, results["objects"])
                    if annotations_container.road_objects is not None:
                        annotations_container.road_objects = self.road_classification_refiner.process(
                            img=img,
                            road_object_box=annotations_container.road_objects,
                            cls_result=cls_result,
                        )

        annotations_container.direction = cast(DirectionBox | None, results["direction"])
        annotations_container.speed = cast(SpeedBox | None, results["speed"])

        return annotations_container
