#
# The RoadObjectExtractor class is responsible for extracting road objects from images
# using a combination of detection and classification models.
#

from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike
from ultralytics.engine.results import Probs, Results  # pyright: ignore[reportMissingTypeStubs]

from configs import globals
from models import ClassificationModel, DetectionModel

from .containers import RoadObjectsBox, Sign, TrafficLight, Vehicle
from .preprocessor import Preprocessor


class RoadObjectExtractor:
    def __init__(self, only_detec_results: bool = False, only_cls_results: bool = False) -> None:
        """
        The RoadObjectExtractor class is responsible for extracting road objects from images
        using a combination of detection and classification models.

        Methods:
            - __init__: Initialize the RoadObjectExtractor.
            - process: Process the input image for road object extraction.
            - refine_classification: Refine the classification of detected road objects.
        """

        self.detection_model: DetectionModel | None = None
        self.classification_model: ClassificationModel | None = None

        if only_detec_results is False:
            self.detection_model = DetectionModel(
                pretrained_model_path=globals.DETECTION_MODEL_PATH,
                device=globals.DETECTION_MODEL_DEVICES,
            )
        if only_cls_results is False:
            self.classification_model = ClassificationModel(
                pretrained_model_path=globals.CLASSIFICATION_MODEL_PATH,
                device=globals.CLASSIFICATION_MODEL_DEVICES,
            )

    def process(
        self, img: MatLike, detect_result: Results | None = None, cls_result: Results | None = None
    ) -> RoadObjectsBox:
        """
        The processing pipeline for road object extraction.

        Args:
            img (MatLike): The input image.
            detect_result (Results | None): The detection results.
            cls_result (Results | None): The classification results.

        Returns:
            RoadObjectsBox: The extracted road objects. Can be empty.
        """

        # ------------------------------ Detection -----------------------------
        if detect_result is not None:
            road_objects_box: RoadObjectsBox = RoadObjectExtractor.processBoxes(result=detect_result)

        elif self.detection_model is not None and detect_result is None:
            working_img: MatLike = img.copy()

            # crop the image to remove the road advisor
            working_img = working_img[globals.ROADOBJECT_EXTRACTION_CROP_TOP :, :, :]

            # preprocess the image
            working_img = Preprocessor.process(img=working_img)
            working_img = cv2.cvtColor(src=working_img, code=cv2.COLOR_BGR2RGB)

            result: Results = self.detection_model.predict(img=working_img)

            road_objects_box: RoadObjectsBox = RoadObjectExtractor.processBoxes(result=result)

        else:
            raise ValueError("No detection results available.")

        # -------------------------- Classification -------------------------------
        # if there are results for given boxes apply them
        if cls_result is not None:
            RoadObjectExtractor.applyRefinedCls(road_objects_box=road_objects_box, result=cls_result)

        # else refine the classification
        elif self.classification_model is not None and cls_result is None:
            for obj in road_objects_box:
                # 0: Sign
                # 1: Traffic-Light
                # 2: Vehicle

                if obj.cls != 2:
                    x1: int = obj.coords[0]
                    y1: int = obj.coords[1]
                    x2: int = obj.coords[2]
                    y2: int = obj.coords[3]
                    cropped_img: MatLike = img[y1:y2, x1:x2]

                    result: Results = self.classification_model.predict(cropped_img)
                    RoadObjectExtractor.applyRefinedCls(road_objects_box=road_objects_box, result=result)
        else:
            raise ValueError("No classification results available.")

        return road_objects_box

    @staticmethod
    def processBoxes(result: Results) -> RoadObjectsBox:
        """
        Create the road objects (vehicles, signs, traffic lights) from prediction.

        Args:
            result (Results): The prediction results.

        Returns:
            RoadObjectsBox: The created road objects.
        """

        road_objects_box: RoadObjectsBox = RoadObjectsBox()

        # Numpy in generell imposes a dynamic typing system so pyright is complaining a lot
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
                road_objects_box.add(road_object=Sign(coords=coords, cls=cls))
            elif cls == 1:
                road_objects_box.add(road_object=TrafficLight(coords=coords, cls=cls))
            elif cls == 2:
                road_objects_box.add(road_object=Vehicle(coords=coords, cls=cls))
            else:
                raise ValueError(f"Unknown class ID: {cls}")

        return road_objects_box

    @staticmethod
    def applyRefinedCls(road_objects_box: RoadObjectsBox, result: Results) -> None:
        """
        Refine the class for sign and traffic light in place.

        Args:
            coords (tuple): The coordinates of the object in the image.

        """

        for obj in road_objects_box:
            if result.boxes is not None:
                # convert tensor to list of tuples
                det_coords = np.round(result.boxes.xyxy.cpu().numpy()).astype(int).tolist()  # type: ignore

                if list(obj.coords) in det_coords:
                    if result.probs is not None:
                        obj.cls = cast(Probs, result.probs).top1
