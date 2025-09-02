import os
from abc import abstractmethod

from cv2.typing import MatLike
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]

from aetd_modules import (
    AnnotationsContainer,
    DirectionExtractor,
    PathPlanner,
    # Pipeline,
    RoadObjectClassificationRefiner,
    RoadObjectDetectionExtractor,
    RoadSegmentsExtractor,
    SpeedDataExtractor,
)
from configs import globals
from models import PreCalculatedLoader


#
# ----------------------------- BASE TAB --------------------------------------
#
class ModulTab(QWidget):
    def __init__(self, parent: QWidget | None = None, table_data: list[tuple[str, str]] | None = None) -> None:
        super().__init__(parent)
        self.img: MatLike | None = None

        # ---------------------- LAYOUT ------------------------
        main_layout = QVBoxLayout(self)

        # ------------------- Enable CheckBox ---------------------
        check_layout = QVBoxLayout()
        self.check_box = QCheckBox(text="Enable")
        self.check_box.setFixedHeight(40)
        self.check_box.setChecked(True)
        check_layout.addWidget(self.check_box, alignment=Qt.AlignmentFlag.AlignTop)
        main_layout.addLayout(check_layout)

        # ---------------------- Table -------------------------

        # if there is table data, create a table
        if table_data is not None:
            # set the basic tabel
            self.info_table = QTableWidget(len(table_data), 2)
            self.info_table.setHorizontalHeaderLabels(["Name", "Value"])
            self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.info_table.verticalHeader().setVisible(False)

            # fill the table from the list
            for row_index, (name, value) in enumerate(table_data):
                self.info_table.setItem(row_index, 0, QTableWidgetItem(name))
                self.info_table.setItem(row_index, 1, QTableWidgetItem(value))

            # hook the table into the layout
            main_layout.addWidget(self.info_table)

        self.setLayout(main_layout)

    @abstractmethod
    def process(self, annotations_container: AnnotationsContainer, to_none: bool = False) -> AnnotationsContainer:
        """
        Default implementation (can be overridden).
        """
        pass


#
# ------------------------------------ TABS ----------------------------------------
#


class DirectionsTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul: DirectionExtractor = DirectionExtractor()

    def process(self, annotations_container: AnnotationsContainer, to_none: bool = False) -> AnnotationsContainer:
        if to_none:
            annotations_container.direction = None
        else:
            annotations_container.direction = self.modul.process(annotations_container.original_img)
        return annotations_container


class SpeedTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul: SpeedDataExtractor = SpeedDataExtractor()

    def process(self, annotations_container: AnnotationsContainer, to_none: bool = False) -> AnnotationsContainer:
        if to_none:
            annotations_container.speed = None
        else:
            annotations_container.speed = self.modul.process(annotations_container.original_img)
        return annotations_container


class RoadObjectDetectionTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        # load the module with results only if path is set
        self.modul: RoadObjectDetectionExtractor = RoadObjectDetectionExtractor(only_detec_results=globals.DET_RESULTS)

        # load the pre-calculated detection results
        if globals.DET_RESULTS:
            path: str = os.path.dirname(globals.DET_RESULTS)
            basename: str = os.path.basename(globals.DET_RESULTS)
            self.det_results: list[tuple[str, Results]] | None = PreCalculatedLoader.load_results(
                input_folder=path, base_name_det=basename
            )["det"]
        else:
            self.det_results = None

        # data to fill into the table: (list of tuples: (name, value))
        table_data: list[tuple[str, str]] = []
        table_data.append(("Precalculated", globals.DET_RESULTS))
        table_data.append(("Model Loaded", str(self.modul.model_loaded())))

        # Note: It is important to load the layout after everything is initialized
        super().__init__(parent, table_data=table_data)

    def process(self, annotations_container: AnnotationsContainer, to_none: bool = False) -> AnnotationsContainer:
        if to_none:
            annotations_container.road_objects = None
        else:
            if self.det_results is not None:
                for img, det in self.det_results:
                    if img == annotations_container.original_img_name:
                        annotations_container.road_objects = self.modul.process(
                            img=annotations_container.original_img, detect_result=det
                        )
                        break
            else:
                annotations_container.road_objects = self.modul.process(img=annotations_container.original_img)

        return annotations_container


class RoadObjectClassificationTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        self.modul: RoadObjectClassificationRefiner = RoadObjectClassificationRefiner(
            only_cls_results=globals.CLS_RESULTS
        )

        # load the pre-calculated classification results
        if globals.CLS_RESULTS:
            path: str = os.path.dirname(globals.CLS_RESULTS)
            basename: str = os.path.basename(globals.CLS_RESULTS)
            self.cls_results: list[tuple[str, Results]] | None = PreCalculatedLoader.load_results(
                input_folder=path, base_name_cls=basename
            )["cls"]
        else:
            self.cls_results = None

        # fill the table
        # data to fill into the table: (list of tuples: (name, value))
        table_data: list[tuple[str, str]] = []
        table_data.append(("Precalculated", globals.CLS_RESULTS))
        table_data.append(("Model Loaded", str(self.modul.model_loaded())))

        # Note: It is important to load the layout after everything is initialized
        super().__init__(parent, table_data=table_data)

    def process(self, annotations_container: AnnotationsContainer, to_none: bool = False) -> AnnotationsContainer:
        if to_none:
            annotations_container.road_objects = None
        else:
            if self.cls_results is not None and annotations_container.road_objects is not None:
                for img, cls in self.cls_results:
                    if img == annotations_container.original_img_name:
                        annotations_container.road_objects = self.modul.process(
                            img=annotations_container.original_img,
                            road_object_box=annotations_container.road_objects,
                            cls_result=cls,
                        )
                    break
            elif annotations_container.road_objects is not None:
                annotations_container.road_objects = self.modul.process(
                    img=annotations_container.original_img,
                    road_object_box=annotations_container.road_objects,
                )

        return annotations_container


class SegmentorTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        # load the module with results only if path is set
        self.modul = RoadSegmentsExtractor(only_results=globals.SEG_RESULTS)

        # load the pre-calculated detection results
        if globals.SEG_RESULTS:
            path: str = os.path.dirname(globals.SEG_RESULTS)
            basename: str = os.path.basename(globals.SEG_RESULTS)
            self.seg_results: list[tuple[str, Results]] | None = PreCalculatedLoader.load_results(
                input_folder=path, base_name_seg=basename
            )["seg"]
        else:
            self.seg_results = None

        # data to fill into the table: (list of tuples: (name, value))
        table_data: list[tuple[str, str]] = []
        table_data.append(("Precalculated", globals.SEG_RESULTS))
        table_data.append(("Model Loaded", str(self.modul.model_loaded())))

        # Note: It is important to load the layout after everything is initialized
        super().__init__(parent, table_data=table_data)

    def process(self, annotations_container: AnnotationsContainer, to_none: bool = False) -> AnnotationsContainer:
        if to_none:
            annotations_container.road_segments = None
        else:
            if self.seg_results is not None:
                for img, seg in self.seg_results:
                    if img == annotations_container.original_img_name:
                        annotations_container.road_segments = self.modul.process(
                            img=annotations_container.original_img, result=seg
                        )
                    break
            else:
                annotations_container.road_segments = self.modul.process(img=annotations_container.original_img)

        return annotations_container


# the path planner needs previously calculated segments to work
class PathTab(ModulTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modul: PathPlanner = PathPlanner()

    def process(self, annotations_container: AnnotationsContainer, to_none: bool = False) -> AnnotationsContainer:
        if to_none:
            annotations_container.road_segments = None
        else:
            if annotations_container.road_segments is not None:
                annotations_container.paths = self.modul.process(
                    road_segment_box=annotations_container.road_segments,
                    width=annotations_container.original_img.shape[1],
                    height=annotations_container.original_img.shape[0],
                )

        return annotations_container


# class FullAnnotationTab(ModulTab):
#     def __init__(self, parent: QWidget | None = None) -> None:
#         # load the module with results only if path is set
#         self.modul = Pipeline(
#             only_cls_results=globals.CLS_RESULTS,
#             only_det_results=globals.DET_RESULTS,
#             only_seg_results=globals.SEG_RESULTS,
#         )

#         path: str = os.path.dirname(globals.SEG_RESULTS)
#         seg_basename: str | None = None
#         cls_basename: str | None = None
#         det_basename: str | None = None

#         # load the pre-calculated detection results
#         if globals.SEG_RESULTS:
#             seg_basename = os.path.basename(globals.SEG_RESULTS)
#         if globals.CLS_RESULTS:
#             cls_basename = os.path.basename(globals.CLS_RESULTS)
#         if globals.DET_RESULTS:
#             det_basename = os.path.basename(globals.DET_RESULTS)

#         self.results: dict[str, list[tuple[str, Results]] | None] = PreCalculatedLoader.load_results(
#             input_folder=path,
#             base_name_det=det_basename,
#             base_name_cls=cls_basename,
#             base_name_seg=seg_basename,
#         )

#         # data to fill into the table: (list of tuples: (name, value))
#         table_data: list[tuple[str, str]] = []
#         table_data.append(("Detection Precalculated", globals.DET_RESULTS))
#         table_data.append(
#             ("Detection Model Loaded", str(self.modul.road_object_detection_extractor.model_loaded())),
#         )
#         table_data.append(("Classification Precalculated", globals.CLS_RESULTS))
#         table_data.append(
#             ("Classification Model Loaded", str(self.modul.road_classification_refiner.model_loaded())),
#         )
#         table_data.append(("Segmentation Precalculated", globals.SEG_RESULTS))
#         table_data.append(
#             ("Segmentation Model Loaded", str(self.modul.road_segments_extractor.model_loaded())),
#         )

#         # Note: It is important to load the layout after everything is initialized
#         super().__init__(parent, table_data=table_data)

#     def process(self, annotations_container: AnnotationsContainer) -> None:
#         if self.results["seg"] is not None:
#             for img, seg in self.results["seg"]:
#                 if img == annotations_container.original_img_name:
#                     self.seg_result: Results = seg
#                     break

#         if self.results["cls"] is not None:
#             for img, cls in self.results["cls"]:
#                 if img == annotations_container.original_img_name:
#                     self.cls_result: Results = cls
#                     break

#         if self.results["det"] is not None:
#             for img, det in self.results["det"]:
#                 if img == annotations_container.original_img_name:
#                     self.det_result: Results = det
#                     break

#         annotations_container = self.modul.process(
#             img=annotations_container.original_img,
#             img_name=annotations_container.original_img_name,
#             cls_result=self.cls_result,
#             detect_result=self.det_result,
#             seg_result=self.seg_result,
#         )

#         self.result_ready.emit(self.annotations_container)
