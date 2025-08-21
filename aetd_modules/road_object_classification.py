#
# The RoadObjectExtractor class is responsible for extracting road objects from images
# using a combination of detection and classification models.
#

from typing import cast

import numpy as np
from cv2.typing import MatLike
from ultralytics.engine.results import Probs, Results  # pyright: ignore[reportMissingTypeStubs]

from configs import globals
from models import ClassificationModel

from .containers import RoadObjectsBox


class RoadObjectClassificationRefiner:
    def __init__(self, only_cls_results: str | bool = False) -> None:
        """
        The RoadObjectClassificationExtractor class is responsible for extracting road objects from images
        using a combination of detection and classification models.

        Methods:
            - __init__: Initialize the RoadObjectClassificationExtractor.
            - process: Process the input image for road object extraction.
            - refine_classification: Refine the classification of detected road objects.
        """

        self.classification_model: ClassificationModel | None = None

        if only_cls_results is False:
            self.classification_model = ClassificationModel(
                pretrained_model_path=globals.CLASSIFICATION_MODEL_PATH,
                device=globals.CLASSIFICATION_MODEL_DEVICES,
            )

    def model_loaded(self) -> bool:
        """
        Return True if the classification model is loaded, False otherwise.
        """
        if self.classification_model is not None:
            return True
        return False

    def process(
        self, img: MatLike, road_object_box: RoadObjectsBox | None, cls_result: Results | None = None
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

        # -------------------------- Classification -------------------------------

        if road_object_box is None:
            raise ValueError("No road objects available for classification.")

        # if there are results for given boxes apply them
        if cls_result is not None:
            self.applyRefinedCls(road_objects_box=road_object_box, result=cls_result)

        # else refine the classification
        elif self.classification_model is not None and cls_result is None:
            for obj in road_object_box:
                # 0: Sign
                # 1: Traffic-Light
                # 2: Vehicle

                if obj.cls != 2:
                    x1: int = obj.coords[0]
                    y1: int = obj.coords[1]
                    x2: int = obj.coords[2]
                    y2: int = obj.coords[3]
                    cropped_img: MatLike = img[y1:y2, x1:x2]

                    result: Results = self.classification_model.predict(img=cropped_img)
                    self.applyRefinedCls(road_objects_box=road_object_box, result=result)
        else:
            raise ValueError("No classification results available.")

        return road_object_box

    def applyRefinedCls(self, road_objects_box: RoadObjectsBox, result: Results) -> None:
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
