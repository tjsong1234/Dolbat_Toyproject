"""Image-only processing; coordinates stay in pixels until planning."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Preprocessed:
    original: np.ndarray
    roi: np.ndarray
    bev: np.ndarray
    binary: np.ndarray
    source_points: np.ndarray


class Preprocessor:
    def __init__(self, config):
        self.cfg = config
        self.size = tuple(config['bev_size'])
        self.matrix = self.inverse = None
        self.input_size = None

    def process(self, frame):
        width = self.cfg['processing_width']
        frame = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])))
        h, w = frame.shape[:2]
        source = np.float32(self.cfg['source_points']) * [w - 1, h - 1]
        if self.input_size != (w, h):
            dest = np.float32(self.cfg['destination_points']) * [self.size[0] - 1, self.size[1] - 1]
            self.matrix = cv2.getPerspectiveTransform(source.astype(np.float32), dest.astype(np.float32))
            self.inverse = np.linalg.inv(self.matrix)
            self.input_size = (w, h)
        roi_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi_mask, [source.astype(np.int32)], 255)
        roi = cv2.bitwise_and(frame, frame, mask=roi_mask)
        bev = cv2.warpPerspective(roi, self.matrix, self.size)
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, self.cfg['white_min_value']),
                            (179, self.cfg['white_max_saturation'], 255))
        # Keep paint adjacent to dark, low-saturation asphalt, excluding green-field symbols.
        road = cv2.inRange(hsv, (0, 0, 20), (179, self.cfg['road_max_saturation'], self.cfg['road_max_value']))
        k = self.cfg['road_neighborhood']
        nearby_road = cv2.dilate(road, np.ones((k, k), np.uint8))
        white = cv2.bitwise_and(white, nearby_road)
        # Remove fine parking-box strokes using local stroke thickness, then restore the edges.
        distance = cv2.distanceTransform(white, cv2.DIST_L2, 3)
        core = np.uint8(distance >= self.cfg['min_half_thickness_px']) * 255
        thick = cv2.bitwise_and(white, cv2.dilate(core, np.ones((5, 5), np.uint8)))
        horizontal = cv2.morphologyEx(thick, cv2.MORPH_OPEN,
                                     np.ones((1, self.cfg['horizontal_kernel_px']), np.uint8))
        binary = cv2.subtract(thick, horizontal)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        clean = np.zeros_like(binary)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area >= self.cfg['min_component_area'] and ch >= 8 and cw < 3.5 * ch:
                clean[labels == i] = 255
        return Preprocessed(frame, roi, bev, clean, source)
