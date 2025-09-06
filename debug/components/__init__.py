from .info_table import InfoTable
from .module_tabs import (
    DirectionsTab,
    # FullAnnotationTab,
    ModulTab,
    PathTab,
    RoadObjectClassificationTab,
    RoadObjectDetectionTab,
    SegmentorTab,
    SpeedTab,
)
from .tab_bars import ModuleTabBar, ViewTabBar
from .view_tabs import ImageViewerTab, VideoViewerTab, View

__all__: list[str] = [
    "ImageViewerTab",
    "ModuleTabBar",
    "ModulTab",
    "DirectionsTab",
    "SpeedTab",
    "RoadObjectClassificationTab",
    "RoadObjectDetectionTab",
    "SegmentorTab",
    "PathTab",
    # "FullAnnotationTab",
    "InfoTable",
    "VideoViewerTab",
    "ViewTabBar",
    "View",
]
