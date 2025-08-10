import cv2
import numpy as np
import math


class PathPlanner:
    def __init__(self):
        pass

    def is_lane_inside_driveable(self, segment_mask, driveable_mask, threshold=0.5):
        intersection = cv2.bitwise_and(driveable_mask, segment_mask)
        lane_area = cv2.countNonZero(segment_mask)
        overlap_area = cv2.countNonZero(intersection)

        # avoid division by zero
        if lane_area == 0:
            return False

        percent_inside = overlap_area / lane_area
        return percent_inside >= threshold

    def mirror_around_x(self, f: np.poly1d, x: float) -> np.poly1d:
        coeffs = -f.coeffs.copy()  # negate all coefficients
        coeffs[-1] += 2 * x  # add width to the constant term
        return np.poly1d(coeffs)

    def getPaths(self, segments):

        driveable = []  # cls == 0
        lanes = []  # cls == 1 or cls == 2

        # 1. sort the segmentations to each type
        for segment in segments:
            if segment.cls == "Driveable":
                driveable.append(segment)
            else:
                lanes.append(segment)

        # 2. we need at least one driveable area
        if len(driveable) < 1:
            return None, None

        # 3. only keep the lanes that are at least 50% in the one of the driveable area polygons
        driveable_masks = [d.mask for d in driveable]
        driveable_mask = np.clip(np.sum(driveable_masks, axis=0), 0, 255).astype(
            np.uint8
        )

        filtered_lanes = [
            l for l in lanes if self.is_lane_inside_driveable(l.mask, driveable_mask)
        ]

        # 4. get the minimum impassable lane from the center for left and right side
        #    and if there is no impassable lane on the left or right side, than set the minimum
        #    to infinity
        maxs = [l.offset for l in filtered_lanes if l.is_left and l.cls == "Impassable"]
        if len(maxs) > 0:
            closest_impassable_left = max(maxs)
        else:
            closest_impassable_left = -math.inf

        mins = [
            l.offset for l in filtered_lanes if l.is_right and l.cls == "Impassable"
        ]
        if len(mins) > 0:
            closest_impassable_right = min(mins)
        else:
            closest_impassable_right = math.inf

        # 5. only keep the left/right lanes that are closer or equal to the center than the closest impassable lane
        #    (so if threre is no impassable lane on that side than keeep all lanes of that side)

        striped_lanes = []
        for l in filtered_lanes:
            if closest_impassable_left <= l.offset <= closest_impassable_right:
                striped_lanes.append(l)

        # 6. sort the lanes by there offset values from left to right, so smallest to biggest x
        sorted_lanes = sorted(striped_lanes, key=lambda l: l.offset)

        # 7. from left to right: calculate a non-linear function that descripts the center of both lanes
        paths = []
        for i in range(len(sorted_lanes) - 1):
            paths.append((sorted_lanes[i].f + sorted_lanes[i + 1].f) / 2)

        # 8. check for only one lane

        if len(sorted_lanes) == 1:
            l = sorted_lanes[0]
            if l.is_left:
                h = self.mirror_around_x(l.f, l.img_width // 2)
                g = (l.f + h) / 2
            else:
                h = self.mirror_around_x(l.f, -l.img_width // 2)
                g = (l.f + h) / 2

            # check if the new path is in the driveable area
            pts = [
                (int(g(y)), y)
                for y in range(l.img_height // 2, l.img_height)
                if 0 <= g(y) < l.img_width
            ]
            pts = np.array(pts, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            g_mask = np.zeros((l.img_height, l.img_width), dtype=np.uint8)
            cv2.fillPoly(g_mask, [pts], 255)
            if self.is_lane_inside_driveable(g_mask, driveable_mask):
                paths.append(g)

        if len(sorted_lanes) == 0:
            paths = None

        # 9. return the driveable paths
        return paths
