#
# The RoadObjectExtractor class is responsible for extracting road objects from images
# using a combination of detection and classification models.
#

from modules.preprocessor import Preprocessor
from ultralytics import YOLO
import cv2
import numpy as np
from configs import debug_default as config
from .data_containers import RoadObjectsBox


class RoadObjectExtractor:
    """
    The RoadObjectExtractor class is responsible for extracting road objects from images
    using a combination of detection and classification models.

    Methods:
        - __init__: Initialize the RoadObjectExtractor.
        - process: Process the input image for road object extraction.
        - refine_classification: Refine the classification of detected road objects.
    """

    def __init__(self) -> None:
        """
        Initialize the RoadObjectExtractor.
        """

        self.img = None
        self.prepro = Preprocessor()
        self.detection_model = YOLO(config.DETECTION_MODEL_PATH)
        self.classification_model = YOLO(config.CLASSIFICATION_MODEL_PATH)
        self.road_objects_box = None

    def process(self, img: np.ndarray) -> None:
        """
        The processing pipeline for road object extraction.

        Args:
            img (np.ndarray): The input image.
        """

        # create a new RoadObjectsBox
        self.road_objects_box = RoadObjectsBox()

        # always create a copy of the original image for safety
        self.img = img.copy()

        # crop the image to remove the road advisor
        self.img = img[config.ROADOBJECT_EXTRACTION_CROP_TOP :, :, :]

        # preprocess the image
        self.img = self.prepro.process(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.predictBoxes()

        return self.road_objects_box

    def predictBoxes(self) -> None:
        """
        Predict road objects in the image (vehicles, signs, traffic lights).
        """

        # make the predictions
        results = self.detection_model.predict(
            self.img,
            device=config.DETECTION_MODEL_DEVICES,
            batch=1,
            verbose=False,
            conf=0.5,
            iou=0.45,
        )

        for result in results:
            xyxy = result.boxes.xyxy  # [x1, y1, x2, y2]
            class_ids = result.boxes.cls.int()

        for coords, cls in zip(xyxy, class_ids):
            # 0: Sign
            # 1: Traffic-Light
            # 2: Vehicle

            if cls == 0:
                # refine the class
                self.road_objects_box.add(
                    Sign(coords, self.refine_classification(coords))
                )
            elif cls == 1:
                # refine the class
                self.road_objects_box.add(
                    TrafficLight(coords, self.refine_classification(coords))
                )
            elif cls == 2:
                self.road_objects_box.add(Vehicle(coords, cls))
            else:
                raise ValueError(f"Unknown class ID: {cls}")

    def refine_classification(self, coords: tuple) -> int:
        """
        Refine the class for sign and traffic light.

        Args:
            coords (tuple): The coordinates of the object in the image.

        Returns:
            int: The refined class id.
        """

        x1, y1, x2, y2 = coords
        cropped_img = self.img[y1:y2, x1:x2]

        # make the prediction
        results = self.classification_model.predict(
            cropped_img,
            device=config.CLASSIFICATION_MODEL_DEVICES,
            batch=1,
            verbose=False,
            conf=0.5,
        )

        return results[0].probs.top1


class Vehicle:
    """
    This class holds the information for detected vehicles.
    """

    def __init__(self, coords, cls) -> None:
        self.coords = coords
        self.cls = cls


class Sign:
    """
    This class holds the information for detected signs.
    """

    def __init__(self, coords, cls) -> None:
        self.coords = coords
        self.cls = cls


class TrafficLight:
    """
    This class holds the information for detected traffic lights.
    """

    def __init__(self, coords, cls) -> None:
        self.coords = coords
        self.cls = cls
