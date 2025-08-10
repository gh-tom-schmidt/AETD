from modules import (
    NavigationDataExtractor,
    RoadObjectDetector,
    RoadSegmentor,
    SpeedDataExtractor,
    PathPlanner,
)


class Unit:
    def __init__(
        self, detection_model_path, classification_model_path, segmentation_model_path
    ):
        self.nde = NavigationDataExtractor()
        self.sde = SpeedDataExtractor()
        self.rod = RoadObjectDetector(detection_model_path, classification_model_path)
        self.rs = RoadSegmentor(segmentation_model_path)
        self.pp = PathPlanner()

    def process(self, img):
        self.img = img.copy()

        self.direction = self.nde.process(img)
        self.speed = self.sde.process(img)
        self.objects = self.rod.process(img)
        self.segments = self.rs.process(img)

        self.paths = self.pp.getPaths(self.segments)

    def getResult(self):
        return self.direction, self.speed, self.objects, self.segments, self.paths
