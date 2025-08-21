from cv2.typing import MatLike
from ultralytics import YOLO  # pyright: ignore[reportMissingTypeStubs]
from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]


class SegmentationModel:
    def __init__(self, pretrained_model_path: str, device: list[int] | str) -> None:
        self.segmentation_model = YOLO(model=pretrained_model_path)
        self.device: list[int] | str = device

    def predict(self, img: MatLike) -> Results:
        """
        Make a prediction on the input image.

        Args:
            img (MatLike): The input image.

        Returns:
            Results: The segmentation results.
        """

        results: list[Results] = self.segmentation_model.predict(  # pyright: ignore[reportUnknownMemberType]
            source=img,
            device=self.device,
            batch=1,
            verbose=False,
            iou=0.45,
            conf=0.6,
        )

        return results[0]

    def batch_predict(self, imgs: list[MatLike]) -> list[Results]:
        """
        Make batch predictions on a list of input images.

        Args:
            imgs (list[MatLike]): The list of input images.

        Returns:
            list[Results]: The list of segmentation results.
        """

        results: list[Results] = self.segmentation_model.predict(  # pyright: ignore[reportUnknownMemberType]
            source=imgs,
            device=self.device,
            batch=1,
            verbose=False,
            iou=0.45,
            conf=0.6,
        )

        return results

    def stream(self) -> None:
        pass
