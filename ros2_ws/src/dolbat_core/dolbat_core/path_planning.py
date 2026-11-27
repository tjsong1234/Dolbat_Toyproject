"""BEV (u right, v down) -> vehicle (X forward, Y left), rear axle origin."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Path:
    pixels: np.ndarray  # Near-to-far [u, v]
    vehicle: np.ndarray  # Near-to-far [X forward, Y left], metres (assumed if uncalibrated)
    lateral_error_m: float  # vehicle minus reference: positive vehicle is left of path
    curvature_per_m: np.ndarray  # positive left curve
    scale_status: str


class PathPlanner:
    def __init__(self, calibration, planning, bev_size):
        self.cfg = calibration
        self.samples = planning['samples']
        self.width, self.height = bev_size
        self.x_m_per_px = calibration['lane_width_m']/calibration['lane_width_px']
        self.y_m_per_px = calibration['forward_span_m']/(self.height-1)
        self.center_u = calibration['vehicle_center_x_normalized']*(self.width-1)

    def pixels_to_vehicle(self, uv):
        uv = np.asarray(uv, dtype=float)
        return np.column_stack((self.cfg['near_distance_m']+(self.height-1-uv[:,1])*self.y_m_per_px,
                                (self.center_u-uv[:,0])*self.x_m_per_px))

    def vehicle_to_pixels(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.column_stack((self.center_u-xy[:,1]/self.x_m_per_px,
                                self.height-1-(xy[:,0]-self.cfg['near_distance_m'])/self.y_m_per_px))

    def plan(self, lanes, max_extrapolation_px=90):
        if not lanes.usable:
            return None
        low = max(0, max(lanes.left.y_min, lanes.right.y_min)-max_extrapolation_px)
        high = min(self.height-1, min(lanes.left.y_max, lanes.right.y_max)+max_extrapolation_px)
        if high-low < 40:
            return None
        v = np.linspace(high, low, self.samples)
        co = (lanes.left.coefficients+lanes.right.coefficients)/2
        uv = np.column_stack((np.polyval(co,v),v))
        xy = self.pixels_to_vehicle(uv)
        # Analytic derivative of Y(X), including unequal x/y pixel scales.
        first = self.x_m_per_px/self.y_m_per_px * (2*co[0]*v+co[1])
        second = -2*co[0]*self.x_m_per_px/self.y_m_per_px**2
        curvature = second/(1+first**2)**1.5
        # Extrapolate only to define signed lateral offset at X=0; expose assumption.
        v0 = self.height-1+self.cfg['near_distance_m']/self.y_m_per_px
        lateral = (float(np.polyval(co,v0))-self.center_u)*self.x_m_per_px
        return Path(uv,xy,lateral,curvature,self.cfg['status'])
