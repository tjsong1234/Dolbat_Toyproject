"""Explicit cm mapping in a mathematical BEV reference frame, not a rear axle."""
from dataclasses import dataclass
import numpy as np


def _trapezoid(y, x):
    """NumPy 1.21 (Humble apt) and 2.x compatible integration."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x)


@dataclass(frozen=True)
class MetricMapping:
    lateral_cm_per_px: float
    forward_cm_per_px: float
    origin_u_px: float
    origin_v_px: float
    status: str = 'course_reference_video_estimate'

    def __post_init__(self):
        vals=[self.lateral_cm_per_px,self.forward_cm_per_px,self.origin_u_px,self.origin_v_px]
        if not np.isfinite(vals).all() or min(vals[:2])<=0:
            raise ValueError('Finite positive scale factors are required')

    def pixels_to_cm(self,uv):
        """Return [X forward, Y left], centimetres. v grows down in the image."""
        uv=np.asarray(uv,dtype=float)
        if uv.ndim!=2 or uv.shape[1]!=2:raise ValueError('Expected N x 2 pixels')
        return np.column_stack(((self.origin_v_px-uv[:,1])*self.forward_cm_per_px,
                                (self.origin_u_px-uv[:,0])*self.lateral_cm_per_px))

    def cm_to_pixels(self,xy):
        xy=np.asarray(xy,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2:raise ValueError('Expected N x 2 metric points')
        return np.column_stack((self.origin_u_px-xy[:,1]/self.lateral_cm_per_px,
                                self.origin_v_px-xy[:,0]/self.forward_cm_per_px))


def physical_arc_length(coefficients,v0,v1,sx,sy):
    """Length of u(v) in cm, correctly accounting for unequal axis scales."""
    vv=np.linspace(v0,v1,129)
    du=2*coefficients[0]*vv+coefficients[1]
    return float(abs(_trapezoid(np.sqrt((sx*du)**2+sy**2),vv)))


def solve_forward_scale(coefficients,v0,v1,sx,length_cm):
    if v1<=v0 or sx<=0 or length_cm<=0:raise ValueError('Invalid arc-length reference')
    if physical_arc_length(coefficients,v0,v1,sx,0)>=length_cm:
        raise ValueError('Lateral displacement already exceeds reference length')
    low,high=0.,max(1.,2*length_cm/(v1-v0))
    for _ in range(60):
        middle=(low+high)/2
        if physical_arc_length(coefficients,v0,v1,sx,middle)<length_cm:low=middle
        else:high=middle
    return (low+high)/2
