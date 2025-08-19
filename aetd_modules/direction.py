#
# The NavigationDataExtractor class is responsible for extracting navigation data from images
# by evaluating the road advisor.
#

from collections.abc import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy import intp

from configs import globals

from .containers import DirectionBox


class DirectionExtractor:
    def __init__(self) -> None:
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

        pass

    @staticmethod
    def process(raw_img: MatLike) -> DirectionBox | None:
        """
        The processing pipeline for extracting navigation data from the image.

        Args:
            raw_img (MatLike): The input image from which to extract navigation data.

        Returns:
            DirectionBox | None: The extracted direction data or None.
        """

        # always create a copy of the original image for safety
        img: MatLike = raw_img.copy()
        img = DirectionExtractor.crop(img=img, height=img.shape[0], width=img.shape[1])

        # take the width and height from the image after the cropping
        height: int = img.shape[0]
        width: int = img.shape[1]

        img = DirectionExtractor.getRedComponents(img=img)
        img = DirectionExtractor.gray(img=img)
        img = DirectionExtractor.binary(img=img)
        img = DirectionExtractor.findContours(img=img)
        weight: int = DirectionExtractor.calculateBias(img=img, width=width, height=height)
        on_lane: float = DirectionExtractor.calculateOnLane(img=img, width=width, height=height)
        direction: int | None = DirectionExtractor.determineDirection(weight=weight, on_lane=on_lane)

        if direction is not None:
            return DirectionBox(direction=direction)
        else:
            return None

    @staticmethod
    def crop(img: MatLike, width: int, height: int) -> MatLike:
        """
        Crop the image to the region of interest defined in the config.

        Args:
            img (MatLike): The input image to crop.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            MatLike: The cropped image.
        """

        top: int = globals.DIRECTION_EXTRACTION_CROP_TOP
        bottom: int = height - globals.DIRECTION_EXTRACTION_CROP_BOTTOM
        left: int = globals.DIRECTION_EXTRACTION_CROP_LEFT
        right: int = width - globals.DIRECTION_EXTRACTION_CROP_RIGHT

        img = img[top:bottom, left:right]
        return img

    @staticmethod
    def getRedComponents(img: MatLike) -> MatLike:
        """
        Enhance the red components (pixels that are mostly red) of the image.

        Args:
            img (MatLike): The input image to process.

        Returns:
            MatLike: The processed image with enhanced red components.
        """

        b: MatLike
        g: MatLike
        r: MatLike
        b, g, r = cv2.split(m=img)

        red_dominant = (r > globals.DIRECTION_EXTRACTION_RED_THRESHOLD) & (r > g) & (r > b)
        img = np.zeros_like(a=img)
        img[red_dominant] = [0, 0, 255]

        return img

    @staticmethod
    def gray(img: MatLike) -> MatLike:
        """
        Convert the image to grayscale.

        Args:
            img (MatLike): The input image to process.

        Returns:
            MatLike: The processed image in grayscale.
        """

        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def binary(img: MatLike) -> MatLike:
        """
        Apply binary thresholding to the image to only get the red parts.

        Args:
            img (MatLike): The input image to process.

        Returns:
            MatLike: The processed image with binary thresholding applied.
        """

        img = cv2.threshold(src=img, thresh=1, maxval=255, type=cv2.THRESH_BINARY)[1]
        return img

    @staticmethod
    def findContours(img: MatLike) -> MatLike:
        """
        Find contours of the red parts in the image.

        Args:
            img (MatLike): The input image to process.

        Returns:
            MatLike: The processed image with contours drawn.
        """

        contours: Sequence[MatLike] = cv2.findContours(
            image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )[0]
        img = np.zeros_like(a=img)
        for cnt in contours:
            area: float = cv2.contourArea(contour=cnt)
            if area >= globals.DIRECTION_EXTRACTION_CONTOUR_MIN_AREA:
                img = cv2.drawContours(
                    image=img,
                    contours=[cnt],
                    contourIdx=-1,
                    color=255,
                    thickness=cv2.FILLED,
                )

        return img

    @staticmethod
    def calculateBias(img: MatLike, width: int, height: int) -> int:
        """
        Get the bias of the detected direction where negative means left and
        positive means right.

        Args:
            img (Img): The input image to process.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            int: The calculated bias.
        """

        center_x: int = width // 2
        top_half: MatLike = img[: height // 2, :]
        xs = np.where(top_half == 255)[1]
        bias: intp = np.sum(a=xs - center_x)
        max_bias: int = len(xs) * center_x
        return int((bias / max_bias) * 100)

    @staticmethod
    def calculateOnLane(img: MatLike, width: int, height: int) -> float:
        """
        Calculate the on-lane ratio based on the detected red parts where
        1 means fully on lane and 0 means fully off lane.

        Args:
            img (Img): The input image to process.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            float: The calculated on-lane ratio.
        """

        pillar: MatLike = img[
            : height // 2,
            globals.DIRECTION_EXTRACTION_CENTER_PILLAR_CROP : width - globals.DIRECTION_EXTRACTION_CENTER_PILLAR_CROP,
        ]
        xs = np.where(pillar == 255)[1]
        return (len(xs) / pillar.size) if len(xs) > 0 else 0

    @staticmethod
    def determineDirection(weight: int, on_lane: float) -> int | None:
        """
        Determine the driving direction based on the weight and on-lane ratio.

        Args:
            weight (int): The calculated weight.
            on_lane (float): The calculated on-lane ratio.

        Returns:
            int | None: The determined driving direction or None.
        """

        if weight <= -40 and on_lane < 0.5:
            return -1
        elif weight >= 40 and on_lane < 0.5:
            return 1
        elif -5 < weight < 5 and on_lane >= 0.5:
            return 0
