#
# The NavigationDataExtractor class is responsible for extracting navigation data from images
# by evaluating the road advisor.
#

import cv2
import numpy as np
from .data_containers import DirectionBox
from configs import debug_default as config


class DirectionExtractor:
    """
    The DirectionExtractor class is responsible for extracting direction data from images
    by evaluating the road advisor.

    Methods:
        - process: The main processing pipeline for extracting direction data.
        - crop: Crop the image to the region of interest.
        - getRedComponents: Enhance the red components of the image.
        - gray: Convert the image to grayscale.
        - binary: Apply binary thresholding to the image.
        - findContours: Find contours of the red parts in the image.
        - calculateBias: Get the bias of the detected direction.
        - calculateOnLane: Calculate the on-lane ratio.
        - determineDirection: Determine the driving direction.

    """

    def __init__(self) -> None:
        """
        Initialize the DirectionExtractor.
        """

        self.img = None
        self.height = None
        self.width = None
        self.direction = None
        self.weight = None
        self.on_lane = None

    def process(self, img: np.ndarray) -> DirectionBox:
        """
        The processing pipeline for extracting navigation data from the image.

        Args:
            img (np.ndarray): The input image from which to extract navigation data.

        Returns:
            DirectionBox: The extracted direction data.
        """

        # always create a copy of the original image for safety
        self.img = img.copy()

        self.crop()

        # take the width and height from the image after the cropping
        self.height, self.width = self.img.shape[:2]

        self.getRedComponents()
        self.gray()
        self.binary()
        self.findContours()
        self.calculateBias()
        self.calculateOnLane()
        self.determineDirection()

        return DirectionBox(self.direction)

    def crop(self) -> None:
        """
        Crop the image to the region of interest defined in the config.
        """

        top = config.DIRECTION_EXTRACTION_CROP_TOP
        bottom = self.img.shape[0] - config.DIRECTION_EXTRACTION_CROP_BOTTOM
        left = config.DIRECTION_EXTRACTION_CROP_LEFT
        right = self.img.shape[1] - config.DIRECTION_EXTRACTION_CROP_RIGHT

        self.img = self.img[top:bottom, left:right]

    def getRedComponents(self) -> None:
        """
        Enhance the red components (pixels that are mostly red) of the image.
        """

        b, g, r = cv2.split(self.img)
        red_dominant = (
            (r > config.DIRECTION_EXTRACTION_RED_THRESHOLD) & (r > g) & (r > b)
        )
        self.img = np.zeros_like(self.img)
        self.img[red_dominant] = [0, 0, 255]

    def gray(self) -> None:
        """
        Convert the image to grayscale.
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def binary(self) -> None:
        """
        Apply binary thresholding to the image to only get the red parts.
        """
        _, self.img = cv2.threshold(self.img, 1, 255, cv2.THRESH_BINARY)

    def findContours(self) -> None:
        """
        Find contours of the red parts in the image.
        """

        contours, _ = cv2.findContours(
            self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.img = np.zeros_like(self.img)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= config.DIRECTION_EXTRACTION_CONTOUR_MIN_AREA:
                self.img = cv2.drawContours(
                    self.img, [cnt], -1, 255, thickness=cv2.FILLED
                )

    def calculateBias(self) -> None:
        """
        Get the bias of the detected direction where negative means left and
        positive means right.
        """

        center_x = self.width // 2
        top_half = self.img[: self.height // 2, :]
        _, xs = np.where(top_half == 255)
        bias = np.sum(xs - center_x)
        max_bias = len(xs) * center_x
        self.weight = int((bias / max_bias) * 100)

    def calculateOnLane(self) -> None:
        """
        Calculate the on-lane ratio based on the detected red parts where
        1 means fully on lane and 0 means fully off lane.
        """

        pillar = self.img[
            : self.height // 2,
            config.DIRECTION_EXTRACTION_CENTER_PILLAR_CROP : self.width
            - config.DIRECTION_EXTRACTION_CENTER_PILLAR_CROP,
        ]
        _, xs = np.where(pillar == 255)
        self.on_lane = (len(xs) / pillar.size) if len(xs) > 0 else 0

    def determineDirection(self) -> None:
        """
        Determine the driving direction based on the weight and on-lane ratio.
        """

        if self.weight <= -40 and self.on_lane < 0.5:
            self.direction = -1
        elif self.weight >= 40 and self.on_lane < 0.5:
            self.direction = 1
        elif -5 < self.weight < 5 and self.on_lane >= 0.5:
            self.direction = 0
        else:
            self.direction = None
