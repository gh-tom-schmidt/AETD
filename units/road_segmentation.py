from units import Preprocessor
from ultralytics import YOLO
import cv2
import numpy as np

class RoadSegmentor:
    def __init__(self, segmentation_model_path):
        self.prepro = Preprocessor()
        self.seg_model_model = YOLO(segmentation_model_path)
        self.segments = Segments()

    def process(self, img):
        # dont use the original image
        self.img = img.copy()

        # preprocess the image
        self.img = self.prepro(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.segmenting()

        return self.segments

    def segmenting(self):
        results = self.seg_model_model.predict(self.img, device='cuda:0', batch=1)
        
        for result in results:
            if result.masks:
                polygons = result.masks.xy                  
                class_ids = result.boxes.cls.int().tolist() 
                confs = result.boxes.conf.tolist()          
                names = result.names                        

                for polygon, cls_id, conf in zip(polygons, class_ids, confs):
                    cls_name = names[cls_id]               
                    pts = polygon.astype(int).reshape((-1, 1, 2))
                    self.segments.add(pts, cls_name, conf, self.img.size)

class Segments:
    def __init__(self):
        self.segments = []
    
    def add(self, pts, cls_name, conf, img_size):
        seg = Segment(pts, cls_name, conf, img_size)
        object = {"points": pts, "cls_name": cls_name, "conf": conf}
        self.segments.append(object)

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
            self.aprox_pts = [(int(self.f(y)), y) for y in range(self.img_height // 2, self.img_height) if 0 <= self.f(y) < self.img_width]
            # get the point where f(height // 4) 
            self.aprox_center_x_pt = int(min(self.aprox_pts, key=lambda pt: abs(pt[1] - int(self.img_height * 0.75)))[0])
            
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
        x_coords = self.pts[:, 0]
        y_coords = self.pts[:, 1]

        coeffs = np.polyfit(y_coords, x_coords, deg=degree)
        f = np.poly1d(coeffs)

        return f

    def mask(self):
        # mask should have only one channel, so a binary img
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.pts], 255)
        
        return mask