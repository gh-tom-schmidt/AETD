import cv2
import numpy as np


class NavigationDataExtractor:
    def __init__(self):
        pass

    def process(self, img):
        # dont use the original image
        self.img = img.copy()

        self.crop()

        self.height, self.width = self.img.shape[:2]

        self.getRedComponents()
        self.gray()
        self.bianry()
        self.findContours()
        self.calculateBias()
        self.calculateOnLane()
        self.determinAdvice()

        return NavData(self.advice)

    def crop(self):
        top = 28
        bottom = self.img.shape[0] - 900
        left = 810
        right = self.img.shape[1] - 810

        self.img = self.img[top:bottom, left:right]

    def getRedComponents(self, threshold=150):
        b, g, r = cv2.split(self.img)
        red_dominant = (r > threshold) & (r > g) & (r > b)
        self.img = np.zeros_like(self.img)
        self.img[red_dominant] = [0, 0, 255]

    def gray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def bianry(self):
        _, self.img = cv2.threshold(self.img, 1, 255, cv2.THRESH_BINARY)

    def findContours(self, min_area=50):
        contours, _ = cv2.findContours(
            self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.img = np.zeros_like(self.img)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                self.img = cv2.drawContours(
                    self.img, [cnt], -1, 255, thickness=cv2.FILLED
                )

    def calculateBias(self):
        center_x = self.width // 2
        top_half = self.img[: self.height // 2, :]
        ys, xs = np.where(top_half == 255)
        offsets = xs - center_x
        bias = np.sum(offsets)
        max_bias = len(xs) * center_x
        self.weight = int((bias / max_bias) * 100)

    def calculateOnLane(self, center_pillar_crop=140):
        pillar = self.img[
            : self.height // 2, center_pillar_crop : self.width - center_pillar_crop
        ]
        ys, xs = np.where(pillar == 255)
        self.on_lane = (len(xs) / pillar.size) if len(xs) > 0 else 0

    def determinAdvice(self):
        if self.weight <= -5 and self.on_lane >= 0.5:
            self.advice = f"Change Lane Left with weight: {self.weight} and centere value: {self.on_lane}"
        elif self.weight >= 5 and self.on_lane >= 0.5:
            self.advice = f"Change Lane Rigth with weight: {self.weight} and centere value: {self.on_lane}"
        elif self.weight <= -40 and self.on_lane < 0.5:
            self.advice = f"Turn Left with weight: {self.weight} and centere value: {self.on_lane}"
        elif self.weight >= 40 and self.on_lane < 0.5:
            self.advice = f"Turn Right with weight: {self.weight} and centere value: {self.on_lane}"
        elif -5 < self.weight < 5 and self.on_lane >= 0.5:
            self.advice = f"Stay on the track with weight: {self.weight} and centere value: {self.on_lane}"
        else:
            self.advice = None


class NavData:
    def __init__(self, advice):
        self.advice = advice
