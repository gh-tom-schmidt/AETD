#
# The RoadSegmentsExtractor class is responsible for extracting road segments from images.
#

from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from ultralytics import YOLO  # pyright: ignore[reportMissingTypeStubs]
from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]

from configs import globals

from .containers import Driveable, Impassable, Passable, Path, RoadSegmentsBox
from .paths import PathExtractor
from .preprocessor import Preprocessor
from .types import Img, ImgT, LImg, LImgT


class RoadSegmentsExtractor:
    def __init__(self) -> None:
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

        self.seg_model_model = YOLO(model=globals.SEGMENTATION_MODEL_PATH)

    def process(self, img: Img) -> RoadSegmentsBox | None:
        """
        The processing pipeline for road segment extraction.
        """

        self.road_segments_box = RoadSegmentsBox()

        # always create a copy of the original image for safety
        self.img: Img = img.copy()
        self.width: int = self.img.shape[1]
        self.height: int = self.img.shape[0]

        # crop the image to remove the road advisor
        self.img = img[globals.ROADSEGMENT_EXTRACTION_CROP_TOP :, :, :]

        # preprocess the image
        self.img = Preprocessor.process(img=self.img)
        self.img = ImgT(img=cv2.cvtColor(src=self.img, code=cv2.COLOR_BGR2RGB))

        self.segmenting()

        return self.road_segments_box

    def segmenting(self) -> None:
        """
        Segment the road into different classes and clean each segment.
        """

        # make the predictions
        results: list[Results] = self.seg_model_model.predict(  # pyright: ignore[reportUnknownMemberType]
            source=self.img,
            device=globals.SEGMENTATION_MODEL_DEVICES,
            batch=1,
            verbose=False,
            iou=0.45,
            conf=0.6,
        )

        for result in results:
            if result.masks is None:
                continue
            polygons: list[NDArray[np.uint8]] = [self.to_numpy_int(r) for r in result.masks.xy]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]

            if result.boxes is None:
                continue
            class_ids: NDArray[np.uint8] = self.to_numpy_int(result.boxes.cls)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

            for poly, cls in zip(polygons, class_ids):
                # reshape the ouput to an opencv format
                pts: NDArray[np.uint8] = poly.astype(np.uint8).reshape((-1, 1, 2))

                # get the cleaned driveable area
                cnt: NDArray[np.uint8] | None = self.findContours(mask=self.morph(mask=self.mask(pts=pts)))

                # create a approximation
                if cnt is None:
                    continue
                path: Path | None = PathExtractor.calculate_path_from_pts(pts=cnt, width=self.width, height=self.height)

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

    def mask(self, pts: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Create a mask for the given polygon points.

        Args:
            pts (NDArray[np.uint8]): The polygon points.

        Returns:
            NDArray[np.uint8]: The mask for the polygon.
        """

        # create blank mask for current polygon
        mask: NDArray[np.uint8] = np.zeros(self.img.shape[:2], dtype=np.uint8)
        # fill it with the segment
        cv2.fillPoly(img=mask, pts=[pts], color=255)

        return mask

    def morph(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Apply morphological operations to clean the mask.

        Args:
            mask (NDArray[np.uint8]): The input mask.

        Returns:
            NDArray[np.uint8]: The cleaned mask.
        """

        kernel: np.ndarray[Any, Any] = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))

        # first opening to remove noise
        opened_mask: np.ndarray[Any, Any] = cv2.morphologyEx(src=mask, op=cv2.MORPH_OPEN, kernel=kernel)

        # then closing to close gaps inside the objects
        cleaned_mask: NDArray[np.uint8] = cast(
            NDArray[np.uint8], cv2.morphologyEx(src=opened_mask, op=cv2.MORPH_CLOSE, kernel=kernel)
        )

        return cleaned_mask

    def findContours(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
        """
        Find contours in the given mask.

        Args:
            mask (NDArray[np.uint8]): The input mask.

        Returns:
            NDArray[np.uint8]: The contours found in the mask.
        """

        # get the contours
        contours: LImg = LImgT(
            img=cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
        )

        if not contours:
            # no contours found
            return None

        # get the biggest contour by area
        cnt: NDArray[np.uint8] = max(contours, key=cv2.contourArea)

        return cnt

    def to_numpy_int(self, x: Tensor | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if isinstance(x, Tensor):
            return x.cpu().numpy().astype(int)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        return np.asarray(x, dtype=int)
