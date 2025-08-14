from modules import (
    NavigationDataExtractor,
    RoadObjectDetector,
    RoadSegmentor,
    SpeedDataExtractor,
    PathPlanner,
)
from concurrent.futures import ThreadPoolExecutor, as_completed


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

        results = {}
        # run all processings in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.nde.process, img): "direction",
                executor.submit(self.sde.process, img): "speed",
                executor.submit(self.rod.process, img): "objects",
                executor.submit(self.rs.process, img): "segments",
            }

            # run the path planer when the segments are ready, even the rest
            # might be still processing
            for future in as_completed(futures):
                key = futures[future]
                results[key] = future.result()
                if key == "segments":
                    self.segments = results["segments"]
                    self.paths = self.pp.getPaths(self.segments)

        self.direction = results["direction"]
        self.speed = results["speed"]
        self.objects = results["objects"]

    def getResult(self):
        return self.direction, self.speed, self.objects, self.segments, self.paths
