#
# The RoadSegmentsExtractor class is responsible for extracting road segments from images.
#

from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]

from configs import globals
from models import SegmentationModel

from .containers import Driveable, Impassable, Passable, Path, RoadSegmentsBox
from .paths import PathExtractor
from .preprocessor import Preprocessor


class RoadSegmentsExtractor:
    def __init__(self, only_results: bool = False) -> None:
        """
        The RoadSegmentsExtractor class is responsible for extracting road segments from images.

        Methods:
            - __init__: Initialize the RoadSegmentsExtractor.
            - process: Process the input image for road segment extraction.
            - segmenting: Segment the road into different classes and clean each segment.
            - mask: Create a mask for the given polygon points.
            - morph: Apply morphological operations to the mask.
            - findContours: Find contours in the given mask.
        """
        self.segmentation_model: SegmentationModel | None = None
        if only_results is False:
            self.segmentation_model = SegmentationModel(
                pretrained_model_path=globals.SEGMENTATION_MODEL_PATH,
                device=globals.SEGMENTATION_MODEL_DEVICES,
            )

    def process(self, img: MatLike, result: Results | None = None) -> RoadSegmentsBox | None:
        """
        The processing pipeline for road segment extraction.
        """

        self.road_segments_box = RoadSegmentsBox()

        self.width: int = img.shape[1]
        self.height: int = img.shape[0]

        # if there is already a result, we can use it
        if result is not None:
            self.segmenting(result=result)

        # if there is a segmentation model but no result, we need to run the model
        elif self.segmentation_model is not None and result is None:
            # always create a copy of the original image for safety
            self.img: MatLike = img.copy()

            # crop the image to remove the road advisor
            self.img = img[globals.ROADSEGMENT_EXTRACTION_CROP_TOP :, :, :]

            # preprocess the image
            self.img = Preprocessor.process(img=self.img)
            self.img = cv2.cvtColor(src=self.img, code=cv2.COLOR_BGR2RGB)

            result = self.segmentation_model.predict(img=self.img)
            self.segmenting(result=result)

        else:
            raise ValueError("Unknown state")

        return self.road_segments_box

    def segmenting(self, result: Results) -> None:
        """
        Segment the road into different classes and clean each segment.
        """

        # Numpy in generell imposes a dynamic typing system so pyright is complaining a lot
        polygons = result.masks.xy  # type: ignore
        class_ids = result.boxes.cls.int().tolist()  # type: ignore

        for poly, cls in zip(polygons, class_ids):  # type: ignore
            # reshape the ouput to an opencv format
            pts: NDArray[np.int32] = poly.astype(np.int32).reshape((-1, 1, 2))

            # get the cleaned driveable area
            cnt: NDArray[np.int32] | None = self.findContours(mask=self.morph(mask=self.mask(pts=pts)))

            # create a approximation
            if cnt is None:
                continue
            path: Path | None = PathExtractor.calculate_path_from_pts(
                pts=cnt.squeeze(1), width=self.width, height=self.height
            )

            # 0: Driveable
            # 1: Passable
            # 2: Impassable

            if path is not None:
                if cls == 0:
                    self.road_segments_box.add(road_segment=Driveable(pts=cnt, path=path))
                elif cls == 1:
                    self.road_segments_box.add(road_segment=Passable(pts=cnt, path=path))
                elif cls == 2:
                    self.road_segments_box.add(road_segment=Impassable(pts=cnt, path=path))
                else:
                    raise ValueError(f"Unknown class ID: {cls}")

    def mask(self, pts: NDArray[np.int32]) -> MatLike:
        """
        Create a mask for the given polygon points.

        Args:
            pts (NDArray[np.int32]): The polygon points.

        Returns:
            MatLike: The mask for the polygon.
        """

        # create blank mask for current polygon
        mask: MatLike = cast(MatLike, np.zeros(self.img.shape[:2], dtype=np.uint8))
        # fill it with the segment
        cv2.fillPoly(img=mask, pts=[pts], color=255)

        return mask

    def morph(self, mask: MatLike) -> MatLike:
        """
        Apply morphological operations to clean the mask.

        Args:
            mask (MatLike): The input mask.

        Returns:
            MatLike: The cleaned mask.
        """

        kernel: MatLike = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))

        # first opening to remove noise
        opened_mask: MatLike = cv2.morphologyEx(src=mask, op=cv2.MORPH_OPEN, kernel=kernel)

        # then closing to close gaps inside the objects
        cleaned_mask: MatLike = cv2.morphologyEx(src=opened_mask, op=cv2.MORPH_CLOSE, kernel=kernel)

        return cast(NDArray[np.uint8], cleaned_mask)

    def findContours(self, mask: MatLike) -> NDArray[np.int32] | None:
        """
        Find contours in the given mask.

        Args:
            mask (MatLike): The input mask.

        Returns:
            NDArray[np.int32]: The contours found in the mask.
        """

        # get the contours
        contours: Sequence[MatLike] = cv2.findContours(
            image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )[0]

        if not contours:
            # no contours founds
            return None

        # get the biggest contour by area
        cnt: MatLike = max(contours, key=cv2.contourArea)

        return cast(NDArray[np.int32], cnt)
