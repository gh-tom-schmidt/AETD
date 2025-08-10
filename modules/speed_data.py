import cv2
import easyocr
from configs.globals import EASYOCR_DEVICES


class SpeedDataExtractor:
    def __init__(self):
        pass

    def process(self, img):
        # dont use the original image
        self.img = img.copy()

        # initilise the text reader
        self.reader = easyocr.Reader(["en"], gpu=EASYOCR_DEVICES)

        self.crop()
        self.gray()
        self.sharpen()
        self.bianry()
        self.read_speed()

        return self.result

    def crop(self):
        top = 5
        bottom = self.img.shape[0] - 1057
        left = 770
        right = self.img.shape[1] - 1125

        self.img = self.img[top:bottom, left:right]

    def gray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def sharpen(self, amount=1.5):
        blurred = cv2.GaussianBlur(self.img, (0, 0), sigmaX=2)
        cv2.addWeighted(self.img, 1 + amount, blurred, -amount, 0)

    def bianry(self):
        _, self.img = cv2.threshold(self.img, 200, 255, cv2.THRESH_BINARY)

    def read_speed(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.result = self.reader.readtext(self.img, detail=0)
        if len(self.result) == 1:
            # only use the first result for now
            self.result = int(self.result[0])
        else:
            self.result = None
