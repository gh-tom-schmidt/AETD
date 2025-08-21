#
# The SpeedDataExtractor class is responsible for extracting speed data from images
# by evaluating the speedometer.
#

from typing import cast

import cv2
import easyocr  # pyright: ignore[reportMissingTypeStubs]
from cv2.typing import MatLike

from configs import globals

from .containers import SpeedBox


class SpeedDataExtractor:
    def __init__(self) -> None:
        """
        The SpeedDataExtractor class is responsible for extracting speed data from images
        by evaluating the speedometer.

        Methods:
            - __init__: Initialize the SpeedDataExtractor.
            - process: Process the input image to extract speed data.
            - crop: Crop the image to the region of interest.
            - gray: Convert the image to grayscale.
            - sharpen: (Un)Sharpen the image.
            - binary: Apply binary thresholding to the image.
            - read_speed: Read the speed from the processed image using OCR.
        """

        self.speed: int | None = None
        self.reader: easyocr.Reader = easyocr.Reader(
            lang_list=["en"],
            gpu=globals.SPEED_EXTRACTION_EASYOCR_DEVICES,
            verbose=False,
        )

    def process(self, img: MatLike) -> SpeedBox | None:
        """
        Processing pipeline for the input image to extract speed data.

        Args:
            img (Img): The input image.

        Returns:
            SpeedBox: The extracted speed data.
        """

        # always create a copy of the original image for safety
        self.img: MatLike = img.copy()

        self.crop()
        self.gray()
        self.sharpen()
        self.binary()
        self.read_speed()

        if self.speed is not None:
            return SpeedBox(speed=self.speed)
        return None

    def crop(self) -> None:
        """
        Crop the image to the region of interest.
        """

        top: int = globals.SPEED_EXTRACTION_CROP_TOP
        bottom: int = self.img.shape[0] - globals.SPEED_EXTRACTION_CROP_BOTTOM
        left: int = globals.SPEED_EXTRACTION_CROP_LEFT
        right: int = self.img.shape[1] - globals.SPEED_EXTRACTION_CROP_RIGHT

        self.img = self.img[top:bottom, left:right]

    def gray(self) -> None:
        """
        Convert the image to grayscale.
        """

        self.img = cv2.cvtColor(src=self.img, code=cv2.COLOR_BGR2GRAY)

    def sharpen(self) -> None:
        """
        (Un)Sharpen the image.
        """

        blurred: MatLike = cv2.GaussianBlur(src=self.img, ksize=(0, 0), sigmaX=2)
        cv2.addWeighted(
            src1=self.img,
            alpha=1 + globals.SPEED_EXTRACTION_SHARPEN_AMOUNT,
            src2=blurred,
            beta=-globals.SPEED_EXTRACTION_SHARPEN_AMOUNT,
            gamma=0,
        )

    def binary(self) -> None:
        """
        Apply binary thresholding to the image.
        """

        self.img = cv2.threshold(src=self.img, thresh=200, maxval=255, type=cv2.THRESH_BINARY)[1]

    def read_speed(self) -> None:
        """
        Read the speed from the processed image using OCR.
        """

        self.img = cv2.cvtColor(src=self.img, code=cv2.COLOR_BGR2RGB)
        self.result: list[str] = cast(list[str], self.reader.readtext(image=self.img, detail=0))  # pyright: ignore[reportUnknownMemberType]
        # make sure that there is only one result
        if len(self.result) == 1:
            try:
                self.speed = int(self.result[0])
            except ValueError:
                self.speed = None
        else:
            self.speed = None
