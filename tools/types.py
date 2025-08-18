from collections.abc import Sequence
from typing import cast

from cv2.typing import NumPyArrayNumeric
from numpy import float32, uint8
from numpy.typing import NDArray

from modules import (
    DirectionBox,
    DirectionExtractor,
    PathPlanner,
    PathsBox,
    Pipeline,
    RoadObjectExtractor,
    RoadObjectsBox,
    RoadSegmentsBox,
    RoadSegmentsExtractor,
    SpeedBox,
    SpeedDataExtractor,
)

Img = NDArray[uint8]
Float = float32
LImg = Sequence[Img]
Box = DirectionBox | SpeedBox | RoadObjectsBox | RoadSegmentsBox | PathsBox | None
Extractors = (
    DirectionExtractor
    | SpeedDataExtractor
    | RoadObjectExtractor
    | RoadSegmentsExtractor
    | PathPlanner
    | Pipeline
    | None
)


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
