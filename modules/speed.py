#
# The SpeedDataExtractor class is responsible for extracting speed data from images
# by evaluating the speedometer.
#

import cv2
import easyocr
from .data_containers import SpeedBox
from configs import debug_default as config
import numpy as np


class SpeedDataExtractor:
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

    def __init__(self) -> None:
        """
        Initialize the SpeedDataExtractor.
        """

        self.img = None
        self.speed = None

    def process(self, img: np.ndarray) -> SpeedBox:
        """
        Processing pipeline for the input image to extract speed data.

        Args:
            img (np.ndarray): The input image.

        Returns:
            SpeedBox: The extracted speed data.
        """

        # always create a copy of the original image for safety
        self.img = img.copy()

        # initilise the text reader
        self.reader = easyocr.Reader(
            ["en"], gpu=config.SPEED_EXTRACTION_EASYOCR_DEVICES
        )

        self.crop()
        self.gray()
        self.sharpen()
        self.binary()
        self.read_speed()

        return SpeedBox(self.speed)

    def crop(self) -> None:
        """
        Crop the image to the region of interest.
        """

        top = config.SPEED_EXTRACTION_CROP_TOP
        bottom = self.img.shape[0] - config.SPEED_EXTRACTION_CROP_BOTTOM
        left = config.SPEED_EXTRACTION_CROP_LEFT
        right = self.img.shape[1] - config.SPEED_EXTRACTION_CROP_RIGHT

        self.img = self.img[top:bottom, left:right]

    def gray(self) -> None:
        """
        Convert the image to grayscale.
        """

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def sharpen(self) -> None:
        """
        (Un)Sharpen the image.
        """

        blurred = cv2.GaussianBlur(self.img, (0, 0), sigmaX=2)
        cv2.addWeighted(
            self.img,
            1 + config.SPEED_EXTRACTION_SHARPEN_AMOUNT,
            blurred,
            -config.SPEED_EXTRACTION_SHARPEN_AMOUNT,
            0,
        )

    def binary(self) -> None:
        """
        Apply binary thresholding to the image.
        """

        _, self.img = cv2.threshold(self.img, 200, 255, cv2.THRESH_BINARY)

    def read_speed(self) -> None:
        """
        Read the speed from the processed image using OCR.
        """

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.result = self.reader.readtext(self.img, detail=0)
        # make sure that there is only one result
        if len(self.result) == 1:
            self.speed = int(self.result[0])
        else:
            self.speed = None
