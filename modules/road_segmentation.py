from modules.preprocessor import Preprocessor
from ultralytics import YOLO
import cv2
import numpy as np
from configs.globals import SEGMENTATION_MODEL_DEVICES
import math


class RoadSegmentor:
    def __init__(self, segmentation_model_path):
        self.prepro = Preprocessor()
        self.seg_model_model = YOLO(segmentation_model_path)
        self.segments = Segments()

    def process(self, img):
        # dont use the original image
        self.img = img.copy()
        # crop by 160 px form the top
        self.img = img[160:, :, :]

        # clear the previous segments
        self.segments.clear()

        # preprocess the image
        self.img = self.prepro.process(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.segmenting()

        return self.segments

    def segmenting(
        self,
        min_conf=0.6,
        min_area_lane=400,
        min_length_lane=300,
        min_area_driveable=1000,
        min_length_driveable=1000,
    ):
        results = self.seg_model_model.predict(
            self.img,
            device=SEGMENTATION_MODEL_DEVICES,
            batch=1,
            verbose=False,
            iou=0.45,
        )

        for result in results:
            if result.masks:
                polygons = result.masks.xy
                class_ids = result.boxes.cls.int().tolist()
                confs = result.boxes.conf.tolist()
                names = result.names

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                height, width = self.img.shape[:2]

                for polygon, cls_id, conf in zip(polygons, class_ids, confs):
                    if conf < min_conf:
                        continue

                    # Create blank mask for current polygon
                    mask = np.zeros((height, width), dtype=np.uint8)

                    pts = polygon.astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)

                    # First opening to remove noise
                    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    # Then closing to close gaps inside the objects
                    cleaned_mask = cv2.morphologyEx(
                        opened_mask, cv2.MORPH_CLOSE, kernel
                    )

                    # Find contours on cleaned mask
                    contours, _ = cv2.findContours(
                        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    cls_name = names[cls_id]

                    for cnt in contours:

                        x_coords = cnt[:, 0, 0]
                        y_coords = cnt[:, 0, 1]
                        horizontal_length = x_coords.max() - x_coords.min()
                        vertical_length = y_coords.max() - y_coords.min()

                        area = cv2.contourArea(cnt)
                        length = cv2.arcLength(cnt, True)

                        if cls_name != "Driveable":
                            if (
                                area > min_area_lane
                                and length > min_length_lane
                                and vertical_length > min_length_lane
                            ):
                                self.segments.add(
                                    cnt.reshape((-1, 1, 2)),
                                    cls_name,
                                    conf,
                                    self.img.shape,
                                )
                        else:
                            if (
                                area > min_area_driveable
                                and length > min_length_driveable
                            ):
                                self.segments.add(
                                    cnt.reshape((-1, 1, 2)),
                                    cls_name,
                                    conf,
                                    self.img.shape,
                                )


class Segments:
    def __init__(self):
        self.segments = []

    def add(self, pts, cls_name, conf, img_size):
        seg = Segment(pts, cls_name, conf, img_size)
        self.segments.append(seg)

    def clear(self):
        self.segments = []

    def __iter__(self):
        return iter(self.segments)


class Segment:
    def __init__(self, pts, cls_name, conf, img_shape):
        self.pts = pts
        self.cls = cls_name
        self.conf = conf
        self.img_shape = img_shape
        self.img_height = img_shape[0]
        self.img_width = img_shape[1]

        self.mask = self.mask()

        # if the class is not the driveable class
        if self.cls != 0:
            # f should be restriced by the y = [0, height]
            self.f = self.approxFunction()

            # approximate points from [height // 2, height] because half the image should be enought to get the lanes for now
            # with [(x, y), ...]
            self.aprox_pts = [
                (int(self.f(y)), y)
                for y in range(self.img_height // 2, self.img_height)
                if 0 <= self.f(y) < self.img_width
            ]

            # get the point where f(height // 4)
            if self.aprox_pts:
                self.aprox_center_x_pt = int(
                    min(
                        self.aprox_pts,
                        key=lambda pt: abs(pt[1] - int(self.img_height * 0.75)),
                    )[0]
                )
            else:
                self.aprox_center_x_pt = -10000

            # if the lane is left from the center than offset is negative else positive
            self.offset = int(self.aprox_center_x_pt - self.img_width // 2)
            self.abs_offset = abs(self.offset)
            self.is_left = True if self.offset <= 0 else False
            self.is_right = True if self.offset > 0 else False
        else:
            self.f = None
            self.offset = None
            self.abs_offset = None
            self.is_left = None
            self.is_right = None

    # this approximated the lane with a function where x = f(y)
    def approxFunction(self, degree=2):
        # Separate into x, y coordinates
        x_coords = self.pts[:, 0, 0]
        y_coords = self.pts[:, 0, 1]

        coeffs = np.polyfit(y_coords, x_coords, deg=degree)
        f = np.poly1d(coeffs)

        return f

    def mask(self):
        # mask should have only one channel, so a binary img
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.pts], 255)

        return mask
