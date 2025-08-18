from collections.abc import Sequence
from typing import cast

from cv2.typing import NumPyArrayNumeric
from numpy import float32, uint8
from numpy.typing import NDArray

Img = NDArray[uint8]
Float = float32
LImg = Sequence[Img]


def ImgT(img: NumPyArrayNumeric) -> Img:
    """
    Convert an opencv NumPyArrayNumeric to Img.
    """
    return cast(Img, img)


def LImgT(img: Sequence[NumPyArrayNumeric]) -> LImg:
    """
    Convert an opencv List of NumPyArrayNumeric to LImg.
    """
    return cast(LImg, img)
