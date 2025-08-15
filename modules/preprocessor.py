#
# The Preprocessor class is responsible for preprocessing images before they are fed into the model.
#

import cv2
import numpy as np
from configs import debug_default as config


class Preprocessor:
    """
    The Preprocessor class is responsible for preprocessing images before they are fed into the model.

    Methods:
        - __init__: Initialize the Preprocessor.
        - process: The main processing pipeline for the image.
        - condCLAHE: Apply CLAHE to the image.
        - gamma: Apply gamma correction to the image.
        - sharpen: Apply sharpening to the image.
    """

    def __init__(self) -> None:
        """
        Initialize the Preprocessor.
        """
        self.img = None

    def process(self, img: np.ndarray) -> np.ndarray:
        """
        The processing pipeline for the image.

        Args:
            img (np.ndarray): The input image to process.

        Returns:
            np.ndarray: The processed image.
        """

        # always create a copy of the original image for safety
        self.img = img.copy()

        self.condCLAHE()
        self.gamma()
        self.sharpen()

        return self.img

    def condCLAHE(self) -> None:
        """
        Apply CLAHE to the image.
        """

        # convert to yuv
        img_yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        luminance = img_yuv[:, :, 0]
        avg_brightness = np.mean(luminance)

        # only apply CLAHE if the average brightness is below the threshold
        if avg_brightness < config.PREPROCESSING_CLAHE_THRESHOLD:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            self.img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def gamma(self) -> None:
        """
        Apply gamma correction to the image.
        """

        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            normalized = i / 255.0
            gamma = 1.0 + (config.PREPROCESSING_GAMMA_MAX - 1.0) * (1 - normalized)
            corrected = pow(normalized, 1.0 / gamma)
            lut[i] = np.clip(corrected * 255, 0, 255)
        self.img = cv2.LUT(self.img, lut)

    def sharpen(self) -> None:
        """
        Apply sharpening to the image.
        """

        blurred = cv2.GaussianBlur(self.img, (0, 0), sigmaX=2)
        cv2.addWeighted(
            self.img,
            1 + config.PREPROCESSING_SHARPEN_AMOUNT,
            blurred,
            -config.PREPROCESSING_SHARPEN_AMOUNT,
            0,
        )
