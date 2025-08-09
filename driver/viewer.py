import cv2
import numpy as np

class DebugView:
    def __init__(self):
        pass
    
    def draw(self, img, results):
        self.direction, self.speed, self.objects, self.segments, self.paths = results

        self.img = img.copy()

        self.draw_segments()
        self.draw_paths()
        self.draw_objects()
        self.draw_info()

        return self.img


    def draw_info(self):
        text = f"{self.direction} | Speed: {self.speed} km/h"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)
        thickness = 2

        h, w = self.img.shape[:2]

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        x = 10
        y = h - 10

        cv2.putText(self.img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    def draw_objects(self):
        for obj in self.objects.objects:
            # object = {"coords": coords, "cls_name": cls_name, "conf": conf}
            # [x1, y1, x2, y2]

            x1, y1, x2, y2 = obj["coords"]

            cv2.rectangle(self.img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_color = (0, 255, 0)
            thickness = 2

            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20  # avoid going above image

            cv2.putText(self.img, obj["cls_name"], (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    def draw_segments(self):
        alpha=0.4
        color = {"Driveable": (0, 255, 255), "Passable": (0, 255, 0), "Impassable": (0, 0, 255)}
        overlay = self.img.copy()

        for segment in self.segments:
            cv2.fillPoly(overlay, [segment.pts], color=color[segment.cls_name])
        
        cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0, dst=self.img)
    
    def draw_paths(self):
        for p in self.paths:    
            pts = [(int(p(y)), y) for y in range(self.img.shape[0] // 2, self.img.shape[0]) if 0 <= p(y) < self.img.shape[1]]
            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.img, [pts_np], isClosed=False, color=(139, 0, 0), thickness=5)


        



