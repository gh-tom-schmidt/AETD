#
# The Preprocessor class is responsible for preprocessing images before they are fed into the model.
#

from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from configs import globals


class Preprocessor:
    def __init__(self) -> None:
        """
        The Preprocessor class is responsible for preprocessing images before they are fed into the model.

        Methods:
            - __init__: Initialize the Preprocessor.
            - process: The main processing pipeline for the image.
            - condCLAHE: Apply CLAHE to the image.
            - gamma: Apply gamma correction to the image.
            - sharpen: Apply sharpening to the image.
        """

        pass

    @staticmethod
    def process(img: MatLike) -> MatLike:
        """
        The processing pipeline for the image.

        Args:
            img (MatLike): The input image to process.

        Returns:
            MatLike: The processed image.
        """

        # always create a copy of the original image for safety
        img = img.copy()

        Preprocessor.condCLAHE(img=img)
        Preprocessor.gamma(img=img)
        Preprocessor.sharpen(img=img)

        return img

    @staticmethod
    def condCLAHE(img: MatLike) -> MatLike:
        """
        Apply CLAHE to the image.
        """

        # convert to yuv
        img_yuv: MatLike = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2YUV)
        luminance: NDArray[np.uint8] = cast(NDArray[np.uint8], img_yuv[:, :, 0])
        avg_brightness: np.floating = np.mean(a=luminance)

        # only apply CLAHE if the average brightness is below the threshold
        if avg_brightness < globals.PREPROCESSING_CLAHE_THRESHOLD:
            clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(src=img_yuv[:, :, 0])
            return cv2.cvtColor(src=img_yuv, code=cv2.COLOR_YUV2BGR)
        else:
            return img

    @staticmethod
    def gamma(img: MatLike) -> MatLike:
        """
        Apply gamma correction to the image.
        """

        lut: NDArray[np.uint8] = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            normalized: float = i / 255.0
            gamma: float = 1.0 + (globals.PREPROCESSING_GAMMA_MAX - 1.0) * (1 - normalized)
            corrected: float = pow(base=normalized, exp=1.0 / gamma)
            lut[i] = np.clip(a=corrected * 255, a_min=0, a_max=255)
        return cv2.LUT(src=img, lut=lut)

    @staticmethod
    def sharpen(img: MatLike) -> MatLike:
        """
        Apply sharpening to the image.
        """

        blurred: MatLike = cv2.GaussianBlur(src=img, ksize=(0, 0), sigmaX=2)
        return cv2.addWeighted(
            src1=img,
            alpha=1 + globals.PREPROCESSING_SHARPEN_AMOUNT,
            src2=blurred,
            beta=-globals.PREPROCESSING_SHARPEN_AMOUNT,
            gamma=0,
        )
