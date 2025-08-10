from modules.preprocessor import Preprocessor
from ultralytics import YOLO
import cv2
from configs.globals import DETECTION_MODEL_DEVICES
from configs.globals import CLASSIFICATION_MODEL_DEVICES

class RoadObjectDetector:
    def __init__(self, detection_model_path, classification_model_path):
        self.prepro = Preprocessor()
        self.detection_model = YOLO(detection_model_path)
        self.road_objects = RoadObjects(classification_model_path)
    
    def process(self, img):
        # dont use the original image
        self.img = img.copy()
        # crop by 160 px form the top
        self.img = img[160:, :, :]

        # preprocess the image
        self.img = self.prepro(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.predictBoxes()

        return self.road_objects
        
    def predictBoxes(self):
        results = self.detection_model.predict(self.img, device=DETECTION_MODEL_DEVICES, batch=1)

        for result in results:
            xyxy = result.boxes.xyxy  # [x1, y1, x2, y2]
            class_ids = result.boxes.cls.int()
            confs = result.boxes.conf
            names = [result.names[cls.item()] for cls in class_ids]

        for coords, name, conf in zip(xyxy, names, confs):
            self.road_objects.add(map(int, coords.tolist()), name, conf)


class RoadObjects():
    def __init__(self, classification_model_path):
        self.objects = []
        self.classification_model = YOLO(classification_model_path)
    
    def add(self, coords, cls_name, conf):
        
        # cls == 2 is "Vehicle"
        if cls_name != "Vehicle": 
            # refine the cls of the sign or light
            results = self.detection_model.predict(self.img, device=CLASSIFICATION_MODEL_DEVICES, batch=1)
            pred = results[0]
            cls_name = pred.names[pred.probs.top1] 
            # get the average confidence from both predictions
            conf = (conf + pred.probs.top1conf.item()) / 2

        object = {"coords": coords, "cls_name": cls_name, "conf": conf}
        self.objects.append(object)