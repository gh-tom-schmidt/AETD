import cv2
import numpy as np


class DebugView:
    def __init__(self, window_name="DebugView", manual_control=False):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        self.manual_control = manual_control

    def window(self, img):
        try:
            cv2.imshow(self.window_name, img)
        except cv2.error:
            # If imshow fails, window probably closed
            return False

        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                # Window closed by user
                return False
        except cv2.error:
            # getWindowProperty failed => window likely closed or not created
            return False

        if self.manual_control:
            print('Waiting for "t" key press to continue or close window to exit...')
            while True:
                key = cv2.waitKey(100) & 0xFF

                # Check if window closed inside loop as well
                try:
                    visible = cv2.getWindowProperty(
                        self.window_name, cv2.WND_PROP_VISIBLE
                    )
                    if visible < 1:
                        print("Window closed by user.")
                        return False
                except cv2.error:
                    return False

                if key == ord("t"):
                    return True
        else:
            cv2.waitKey(1)
            return True

    def close(self):
        cv2.destroyWindow(self.window_name)

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
        color = (255, 255, 0)
        thickness = 2

        h, w = self.img.shape[:2]

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        x = 10
        y = h - 10

        cv2.putText(
            self.img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
        )

    def draw_objects(self):
        for obj in self.objects.objects:
            # object = {"coords": coords, "cls_name": cls_name, "conf": conf}
            # [x1, y1, x2, y2]
            x1, y1, x2, y2 = obj["coords"]

            # take the offset of 160px into account
            y1 += 160
            y2 += 160

            cv2.rectangle(self.img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_color = (0, 255, 0)
            thickness = 2

            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20  # avoid going above image

            cv2.putText(
                self.img,
                obj["cls_name"],
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    def draw_segments(self):
        alpha = 0.4
        color = {
            "Driveable": (0, 255, 255),
            "Passable": (0, 255, 0),
            "Impassable": (0, 0, 255),
        }
        overlay = self.img.copy()

        # draw the driveable first
        for segment in self.segments:
            if segment.cls == "Driveable":
                # take the cropping into account
                pts = segment.pts.copy()
                pts[:, 0, 1] += 160

                cv2.fillPoly(overlay, [pts], color=color[segment.cls])

        for segment in self.segments:
            if segment.cls != "Driveable":
                # take the cropping into account
                pts = segment.pts.copy()
                pts[:, 0, 1] += 160

                cv2.fillPoly(overlay, [pts], color=color[segment.cls])

        cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0, dst=self.img)

    def draw_paths(self):
        y_offset = 160
        height = self.img.shape[0]

        for p in self.paths:
            pts = []

            for y in range(height // 2 - y_offset, height - y_offset):
                x = p(y)
                if 0 <= x < self.img.shape[1]:
                    y_orig = y + y_offset
                    pts.append((int(x), y_orig))

            if pts:
                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    self.img, [pts_np], isClosed=False, color=(139, 0, 0), thickness=5
                )
