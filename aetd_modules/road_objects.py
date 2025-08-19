#
# The RoadObjectExtractor class is responsible for extracting road objects from images
# using a combination of detection and classification models.
#

from typing import cast

import cv2
from cv2.typing import MatLike
from ultralytics import YOLO  # pyright: ignore[reportMissingTypeStubs]
from ultralytics.engine.results import Probs, Results  # pyright: ignore[reportMissingTypeStubs]

from configs import globals

from .containers import RoadObjectsBox, Sign, TrafficLight, Vehicle
from .preprocessor import Preprocessor


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

    def process(self, img: MatLike) -> RoadObjectsBox | None:
        """
        The processing pipeline for road object extraction.

        Args:
            img (np.ndarray): The input image.
        """

        # create a new RoadObjectsBox
        self.road_objects_box: RoadObjectsBox = RoadObjectsBox()

        # always create a copy of the original image for safety
        self.img: MatLike = img.copy()

        # crop the image to remove the road advisor
        self.img = img[globals.ROADOBJECT_EXTRACTION_CROP_TOP :, :, :]

        # preprocess the image
        self.img = Preprocessor.process(img=self.img)
        self.img = cv2.cvtColor(src=self.img, code=cv2.COLOR_BGR2RGB)

        self.predictBoxes()

        return self.road_objects_box

    def predictBoxes(self) -> None:
        """
        Predict road objects in the image (vehicles, signs, traffic lights).
        """

        results: list[Results] = self.detection_model.predict(  # pyright: ignore[reportUnknownMemberType]
            source=self.img,
            device=globals.DETECTION_MODEL_DEVICES,
            batch=1,
            verbose=False,
            conf=0.5,
            iou=0.45,
        )

        # Numpy in generell imposes a dynamic typing system so pyright is complaining a lot
        for result in results:
            # Enforce a list of tuples with the coordinated in it [x1, y1, x2, y2]
            # where each value must be an integer, because coordinates are expected to be integers
            xyxy: list[tuple[int, int, int, int]] = [
                (int(x1), int(y1), int(x2), int(y2))  # type: ignore
                for x1, y1, x2, y2 in result.boxes.xyxy.tolist()  # type: ignore
            ]
            class_ids = result.boxes.cls.int().tolist()  # type: ignore

            for coords, cls in zip(xyxy, class_ids):  # type: ignore
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

                    self.road_objects_box.add(road_object=TrafficLight(coords=coords, cls=cls))

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
        cropped_img: MatLike = self.img[y1:y2, x1:x2]

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
