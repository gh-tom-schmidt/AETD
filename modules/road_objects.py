#
# The RoadObjectExtractor class is responsible for extracting road objects from images
# using a combination of detection and classification models.
#

from typing import Any, cast

import cv2
import numpy as np
from ultralytics import YOLO  # pyright: ignore[reportMissingTypeStubs]
from ultralytics.engine.results import Probs, Results  # pyright: ignore[reportMissingTypeStubs]

from configs import globals
from modules.preprocessor import Preprocessor
from tools import Img, ImgT

from .data_containers import RoadObjectsBox


class RoadObjectExtractor:
    def __init__(self) -> None:
        """
        The RoadObjectExtractor class is responsible for extracting road objects from images
        using a combination of detection and classification models.

        Methods:
            - __init__: Initialize the RoadObjectExtractor.
            - process: Process the input image for road object extraction.
            - refine_classification: Refine the classification of detected road objects.
        """

        self.detection_model: YOLO = YOLO(globals.DETECTION_MODEL_PATH)
        self.classification_model: YOLO = YOLO(globals.CLASSIFICATION_MODEL_PATH)

    def process(self, img: Img) -> RoadObjectsBox | None:
        """
        The processing pipeline for road object extraction.

        Args:
            img (np.ndarray): The input image.
        """

        # create a new RoadObjectsBox
        self.road_objects_box: RoadObjectsBox = RoadObjectsBox()

        # always create a copy of the original image for safety
        self.img: Img = img.copy()

        # crop the image to remove the road advisor
        self.img = img[globals.ROADOBJECT_EXTRACTION_CROP_TOP :, :, :]

        # preprocess the image
        self.img = Preprocessor.process(img=self.img)
        self.img = ImgT(img=cv2.cvtColor(src=self.img, code=cv2.COLOR_BGR2RGB))

        self.predictBoxes()

        return self.road_objects_box

    def predictBoxes(self) -> None:
        """
        Predict road objects in the image (vehicles, signs, traffic lights).
        """

        # make the predictions
        results: list[Results] = self.detection_model.predict(  # pyright: ignore[reportUnknownMemberType]
            source=self.img,
            device=globals.DETECTION_MODEL_DEVICES,
            batch=1,
            verbose=False,
            conf=0.5,
            iou=0.45,
        )

        for result in results:
            # [x1, y1, x2, y2]
            if result.boxes is None:
                continue
            xyxy: np.ndarray[Any, Any] = cast(np.ndarray[Any, Any], result.boxes.xyxy)  # pyright: ignore[reportUnknownMemberType]
            class_ids: np.ndarray[Any, Any] = cast(np.ndarray[Any, Any], result.boxes.cls).astype(dtype=int)  # pyright: ignore[reportUnknownMemberType]

            for coords, cls in zip(xyxy, class_ids):
                # 0: Sign
                # 1: Traffic-Light
                # 2: Vehicle

                if cls == 0:
                    # refine the class if possible
                    refined_cls: int | None = self.refine_classification(coords=coords)
                    if refined_cls is not None:
                        cls: int = refined_cls

                    self.road_objects_box.add(road_object=Sign(coords=coords, cls=cls))

                elif cls == 1:
                    refined_cls: int | None = self.refine_classification(coords=coords)
                    if refined_cls is not None:
                        cls: int = refined_cls

                    self.road_objects_box.add(road_object=Sign(coords=coords, cls=cls))

                elif cls == 2:
                    self.road_objects_box.add(road_object=Vehicle(coords=coords, cls=cls))

                else:
                    raise ValueError(f"Unknown class ID: {cls}")

    def refine_classification(self, coords: tuple[int, int, int, int]) -> int | None:
        """
        Refine the class for sign and traffic light.

        Args:
            coords (tuple): The coordinates of the object in the image.

        Returns:
            int: The refined class id.
        """
        x1: int = coords[0]
        y1: int = coords[1]
        x2: int = coords[2]
        y2: int = coords[3]
        cropped_img: Img = self.img[y1:y2, x1:x2]

        # make the prediction
        results: list[Results] = self.classification_model.predict(  # pyright: ignore[reportUnknownMemberType]
            source=cropped_img,
            device=globals.CLASSIFICATION_MODEL_DEVICES,
            batch=1,
            verbose=False,
            conf=0.5,
        )
        if results[0].probs is not None:
            return cast(Probs, results[0].probs).top1
        else:
            return None


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
