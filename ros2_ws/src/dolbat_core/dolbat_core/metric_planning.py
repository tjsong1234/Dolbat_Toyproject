"""Week 3: observed pixels -> cm lane functions -> a continuous planned path.

The origin is an explicit BEV reference. No rear-axle location, steering command or
closed-loop motion is inferred from this recorded camera video.
"""
from dataclasses import dataclass
import numpy as np
from .metric_mapping import MetricMapping


@dataclass
class MetricBoundary:
    coefficients: np.ndarray  # Y_left_cm = A*X_forward_cm**2 + B*X_forward_cm + C
    points_cm: np.ndarray
    row_centers_cm: np.ndarray
    inlier_rows: np.ndarray
    interval_cm: tuple
    rmse_cm: float
    all_rows_rmse_cm: float

    def y(self,x):return np.polyval(self.coefficients,x)
    def slope(self,x):return np.polyval(np.polyder(self.coefficients),x)
    def curvature(self,x):
        return np.polyval(np.polyder(self.coefficients,2),x)/(1+self.slope(x)**2)**1.5


@dataclass
class MetricPlan:
    status: str
    reason: str
    left: MetricBoundary | None
    right: MetricBoundary | None
    path_cm: np.ndarray
    path_uv: np.ndarray
    center_coefficients: np.ndarray | None
    inferred_boundary_cm: np.ndarray | None
    curvature_per_cm: np.ndarray
    method: str

    @property
    def usable(self):return len(self.path_cm)>=2


def fit_metric_boundary(lane,mapping,config,windows=16,height=640):
    if lane is None or lane.inferred or len(lane.pixels)<3:return None
    px=np.asarray(lane.pixels,dtype=float);v=px[:,1].astype(int)
    counts=np.bincount(v,minlength=height);sums=np.bincount(v,weights=px[:,0],minlength=height)
    rows=np.flatnonzero(counts);uv=np.column_stack((sums[rows]/counts[rows],rows))
    xy=mapping.pixels_to_cm(uv);order=np.argsort(xy[:,0]);xy=xy[order];rows=rows[order]
    if len(xy)<12 or np.ptp(xy[:,0])<1:return None
    x,y=xy.T;x0=float(x.mean());scale=max(float(np.std(x)),1.)
    z=(x-x0)/scale;keep=np.ones(len(x),bool)
    bins=np.minimum(rows*windows//height,windows-1)
    weights=1/np.sqrt(np.bincount(bins,minlength=windows)[bins])
    for _ in range(config['robust_iterations']):
        if keep.sum()<12:return None
        cf=np.polyfit(z[keep],y[keep],config['degree'],w=weights[keep])
        error=y-np.polyval(cf,z);mad=float(np.median(abs(error[keep]-np.median(error[keep]))))
        limit=max(2*mapping.lateral_cm_per_px,min(config['fit_outlier_limit_cm'],3*1.4826*mad))
        keep=abs(error)<=limit
    if keep.sum()<12:return None
    cf=np.polyfit(z[keep],y[keep],config['degree'],w=weights[keep])
    co=np.poly1d(cf)(np.poly1d([1/scale,-x0/scale])).c
    error=y-np.polyval(co,x)
    return MetricBoundary(co,mapping.pixels_to_cm(px),xy,keep,(float(x[keep].min()),float(x[keep].max())),
                          float(np.sqrt(np.mean(error[keep]**2))),float(np.sqrt(np.mean(error**2))))


def normal_offset(boundary,x,distance_cm):
    """Continuous left-normal offset: (t-d*f'/q, f(t)+d/q), q=sqrt(1+f'^2)."""
    slope=boundary.slope(x);q=np.sqrt(1+slope**2)
    return np.column_stack((x-distance_cm*slope/q,boundary.y(x)+distance_cm/q))


class MetricLanePlanner:
    def __init__(self,mapping,config,lane_center_width_cm,windows=16,bev_height=640):
        if lane_center_width_cm<=0:raise ValueError('Positive centerline lane width required')
        if config['degree']!=2:raise ValueError('The verified week-3 model uses quadratic boundaries')
        self.mapping=mapping;self.cfg=config;self.width=lane_center_width_cm
        self.windows=windows;self.height=bev_height

    def plan(self,lanes):
        empty=np.empty((0,2));no=lambda status,reason,l=None,r=None:MetricPlan(status,reason,l,r,empty,empty,None,None,np.array([]),'none')
        if not lanes.usable:return no('NO_OBSERVATION','perception did not return a usable observation')
        left=fit_metric_boundary(lanes.left,self.mapping,self.cfg,self.windows,self.height)
        right=fit_metric_boundary(lanes.right,self.mapping,self.cfg,self.windows,self.height)
        n=self.cfg['samples'];minimum=self.cfg['min_path_span_cm']
        if left is not None and right is not None:
            lo=max(left.interval_cm[0],right.interval_cm[0],0);hi=min(left.interval_cm[1],right.interval_cm[1])
            if hi-lo<minimum:return no('NO_SHARED_RANGE','observed metric intervals do not overlap sufficiently',left,right)
            xx=np.linspace(lo,hi,n);co=(left.coefficients+right.coefficients)/2
            if np.any(left.y(xx)<=right.y(xx)):return no('CROSSED_BOUNDARIES','observed metric lane functions cross',left,right)
            path=np.column_stack((xx,np.polyval(co,xx)))
            dy=np.polyval(np.polyder(co),xx);curvature=np.polyval(np.polyder(co,2),xx)/(1+dy**2)**1.5
            return MetricPlan('OBSERVED_PAIR','midpoint of two metric quadratic functions over their shared observed range',left,right,
                path,self.mapping.cm_to_pixels(path),co,None,curvature,'midpoint_quadratic')
        boundary=left if left is not None else right
        if boundary is None:return no('NO_METRIC_FIT','too few valid observed metric points')
        lo,hi=boundary.interval_cm
        if hi-lo<minimum:return no('TOO_SHORT','observed metric interval too short',left,right)
        x=np.linspace(lo,hi,n);d=-self.width/2 if left is not None else self.width/2
        path=normal_offset(boundary,x,d);factor=1-d*boundary.curvature(x)
        valid=np.isfinite(path).all(axis=1)&(path[:,0]>=0)&(factor>self.cfg['offset_singularity_margin'])
        valid&=np.gradient(path[:,0])>0
        ids=np.flatnonzero(valid);groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
        groups=[group for group in groups if len(group)>1 and np.ptp(path[group,0])>=minimum]
        if not groups:return no('INVALID_OFFSET','normal-offset path folds or has insufficient forward range',left,right)
        group=max(groups,key=lambda g:np.ptp(path[g,0]));truncated=len(group)<len(path)
        path=path[group];x=x[group];factor=factor[group]
        partner=normal_offset(boundary,x,2*d)
        if np.any(1-2*d*boundary.curvature(x)<=self.cfg['offset_singularity_margin']):partner=None
        curvature=boundary.curvature(x)/factor
        status='ESTIMATED_FROM_LEFT' if left is not None else 'ESTIMATED_FROM_RIGHT'
        if truncated:status+='_TRUNCATED'
        return MetricPlan(status,'one observed metric function; center uses known lane-width normal offset',left,right,
            path,self.mapping.cm_to_pixels(path),None,partner,curvature,'normal_offset_parametric')
