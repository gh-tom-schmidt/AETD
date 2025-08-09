import cv2
import numpy as np

class Preprocessor():
    def __init__(self):
        pass

    def process(self, img):
         # dont use the original image
        self.img = img.copy() 

        self.condCLAHE()
        self.gamma()
        self.sharpen()

        return self.img

    def condCLAHE(self, threshold=50):
        img_yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        luminance = img_yuv[:, :, 0]
        avg_brightness = np.mean(luminance)

        if avg_brightness < threshold:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            self.img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    def gamma(self, max=1.5):
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            normalized = i / 255.0
            gamma = 1.0 + (max - 1.0) * (1 - normalized)  
            corrected = pow(normalized, 1.0 / gamma)
            lut[i] = np.clip(corrected * 255, 0, 255)
        self.img = cv2.LUT(self.img, lut)
    
    def sharpen(self, amount=1.5):
        blurred = cv2.GaussianBlur(self.img, (0, 0), sigmaX=2)
        cv2.addWeighted(self.img, 1 + amount, blurred, -amount, 0)

