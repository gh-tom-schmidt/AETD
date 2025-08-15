#
# The RoadSegmentsExtractor class is responsible for extracting road segments from images.
#

from modules.preprocessor import Preprocessor
from ultralytics import YOLO
import cv2
import numpy as np
import numpy as np
from configs import debug_default as config
from .data_containers import RoadSegmentsBox
from .paths import Path, PathExtractor


class RoadSegmentsExtractor:
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

    def __init__(self) -> None:
        """
        Initialize the RoadSegmentsExtractor.
        """

        self.img = None
        self.prepro = Preprocessor()
        self.seg_model_model = YOLO(config.SEGMENTATION_MODEL_PATH)
        self.road_segments_box = None

    def process(self, img: np.ndarray) -> None:
        """
        The processing pipeline for road segment extraction.
        """

        self.road_segments_box = RoadSegmentsBox()

        # always create a copy of the original image for safety
        self.img = img.copy()

        # crop the image to remove the road advisor
        self.img = img[config.ROADSEGMENT_EXTRACTION_CROP_TOP :, :, :]

        # preprocess the image
        self.img = self.prepro.process(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.segmenting()

        return self.road_segments_box

    def segmenting(self) -> None:
        """
        Segment the road into different classes and clean each segment.
        """

        # make the predictions
        results = self.seg_model_model.predict(
            self.img,
            device=config.SEGMENTATION_MODEL_DEVICES,
            batch=1,
            verbose=False,
            iou=0.45,
            conf=0.6,
        )

        for result in results:
            if result.masks:
                polygons = result.masks.xy
                class_ids = result.boxes.cls.int().tolist()

                for poly, cls in zip(polygons, class_ids):

                    # reshape the ouput to an opencv format
                    pts = poly.astype(np.int32).reshape((-1, 1, 2))

                    # get the cleaned driveable area
                    cnt = self.findContours(self.morph(self.mask(pts)))

                    # create a approximation
                    path = PathExtractor.calculate_path_from_pts(
                        cnt, self.img.shape[:2]
                    )

                    # 0: Driveable
                    # 1: Passable
                    # 2: Impassable

                    if cls == 0:
                        self.road_segments_box.add(Driveable(cnt, path))
                    elif cls == 1:
                        self.road_segments_box.add(Passable(cnt, path))
                    elif cls == 2:
                        self.road_segments_box.add(Impassable(cnt, path))
                    else:
                        raise ValueError(f"Unknown class ID: {cls}")

    def mask(self, pts: np.ndarray) -> np.ndarray:
        """
        Create a mask for the given polygon points.

        Args:
            pts (np.ndarray): The polygon points.

        Returns:
            np.ndarray: The mask for the polygon.
        """

        # create blank mask for current polygon
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        # fill it with the segment
        cv2.fillPoly(mask, [pts], 255)

        return mask

    def morph(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean the mask.

        Args:
            mask (np.ndarray): The input mask.

        Returns:
            np.ndarray: The cleaned mask.
        """

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # first opening to remove noise
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # then closing to close gaps inside the objects
        cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

        return cleaned_mask

    def findContours(self, mask: np.ndarray) -> np.ndarray:
        """
        Find contours in the given mask.

        Args:
            mask (np.ndarray): The input mask.

        Returns:
            np.ndarray: The contours found in the mask.
        """

        # get the contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # no contours found
            return None

        # get the biggest contour by area
        cnt = max(contours, key=cv2.contourArea)

        return cnt


class Driveable:
    """
    This class holds the information for a driveable area.
    """

    def __init__(self, pts: np.ndarray, path: Path) -> None:
        self.pts = pts
        self.path = path


class Impassable:
    """
    This class holds the information for a impassable lane.
    """

    def __init__(self, pts: np.ndarray, path: Path) -> None:
        self.pts = pts
        self.path = path


class Passable:
    """
    This class holds the information for a passable lane.
    """

    def __init__(self, pts: np.ndarray, path: Path) -> None:
        self.pts = pts
        self.path = path
