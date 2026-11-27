"""Bounded path stabilization without extending the current observation range.

The boundary fits and their observed/inferred status stay unchanged. Only the
planned path is blended with the previous path where BOTH frames have a valid
forward range. This is a short temporal filter, not ego-motion compensation.
"""
from dataclasses import dataclass,field,fields
import numpy as np
from .metric_planning import MetricPlan,MetricLanePlanner


@dataclass
class StableMetricPlan(MetricPlan):
    raw_path_cm:np.ndarray=field(default_factory=lambda:np.empty((0,2)))
    raw_curvature_per_cm:np.ndarray=field(default_factory=lambda:np.array([]))
    raw_center_coefficients:np.ndarray|None=None
    stabilization_applied:bool=False
    max_path_adjustment_cm:float=0.
    stabilization_previous_weight:float=0.
    shared_range_cm:tuple|None=None
    stabilization_reset_reason:str=''
    stabilization_context:dict=field(default_factory=dict)
    stabilization_previous_context:dict|None=None
    stabilization_range_jump_cm:float=0.


def sampled_curvature(path_cm):
    """Signed d²Y/dX² / (1+(dY/dX)²)^1.5, with X and Y both in cm."""
    points=np.asarray(path_cm,dtype=float)
    if len(points)<3:return np.zeros(len(points))
    if not np.isfinite(points).all() or np.any(np.diff(points[:,0])<=0):
        raise ValueError('Finite, strictly increasing forward path coordinates required')
    dy=np.gradient(points[:,1],points[:,0],edge_order=2)
    ddy=np.gradient(dy,points[:,0],edge_order=2)
    return ddy/(1+dy*dy)**1.5


class StableMetricLanePlanner(MetricLanePlanner):
    """Use current path evidence, with a configurable maximum lateral correction.

    Defaults at 24 FPS: time constant 0.075 s, at most 1 cm from the CURRENT raw
    path. Newly exposed ranges receive current raw points, never extrapolation.
    """
    def __init__(self,mapping,config,lane_center_width_cm,windows=16,bev_height=640,fps=24.):
        super().__init__(mapping,config,lane_center_width_cm,windows,bev_height)
        self.fps=float(fps)
        self.tau=float(config.get('path_smoothing_time_constant_s',.075))
        self.limit=float(config.get('max_path_adjustment_cm',1.))
        self.edge=float(config.get('path_blend_edge_cm',100.))
        self.min_shared=float(config.get('path_min_shared_span_cm',20.))
        self.pair_range_reset=float(config.get('path_pair_range_reset_cm',20.))
        if not np.isfinite([self.fps,self.tau,self.limit,self.edge,self.min_shared,self.pair_range_reset]).all():
            raise ValueError('Finite path stabilization settings required')
        if self.fps<=0 or min(self.tau,self.limit,self.edge,self.min_shared,self.pair_range_reset)<0:
            raise ValueError('Positive FPS and nonnegative path stabilization settings required')
        self.previous_path_cm=None;self.previous_status=None;self.previous_raw_range_cm=None
        self.previous_context=None

    def plan(self,lanes):
        return self.stabilize(super().plan(lanes))

    @staticmethod
    def observation_context(raw,context=None):
        """Normalize current geometric evidence independently of display labels.

        Older planners supply MetricBoundary objects directly, so callers that
        do not pass a context still get an evidence-derived default. The yellow
        guided planner can explicitly identify its reference side and the
        support interval used by its completion model before stabilization.
        """
        left,right=raw.left,raw.right
        both=left is not None and right is not None
        interval=None
        if both:
            lo=max(left.interval_cm[0],right.interval_cm[0])
            hi=min(left.interval_cm[1],right.interval_cm[1])
            if hi>=lo:interval=(float(lo),float(hi))
        elif left is not None or right is not None:
            interval=tuple((left if left is not None else right).interval_cm)
        derived={'reference_side':'right' if right is not None else 'left' if left is not None else 'none',
            'both_observed':both,'observation_range_cm':interval}
        if context is not None:
            if not isinstance(context,dict):raise ValueError('Observation context must be a dictionary')
            derived.update(context)
        side=derived['reference_side']
        if side not in ('left','right','none'):
            raise ValueError('Observation reference side must be left, right or none')
        interval=derived['observation_range_cm']
        if interval is not None:
            values=np.asarray(interval,dtype=float)
            if values.shape!=(2,) or not np.isfinite(values).all() or values[1]<values[0]:
                raise ValueError('Finite, ordered observation support interval required')
            interval=(float(values[0]),float(values[1]))
        return {'reference_side':side,'both_observed':bool(derived['both_observed']),
            'observation_range_cm':interval}

    def stabilize(self,raw,context=None):
        """Keep raw fits, metadata and range; clear all history on missing paths."""
        values={f.name:getattr(raw,f.name) for f in fields(MetricPlan)}
        raw_path=np.asarray(raw.path_cm,dtype=float).copy()
        current_context=self.observation_context(raw,context)
        result=StableMetricPlan(**values,raw_path_cm=raw_path,
            raw_curvature_per_cm=np.asarray(raw.curvature_per_cm).copy(),
            raw_center_coefficients=None if raw.center_coefficients is None else raw.center_coefficients.copy(),
            stabilization_context=current_context.copy(),
            stabilization_previous_context=None if self.previous_context is None else self.previous_context.copy())
        if not raw.usable:
            self.previous_path_cm=None;self.previous_status=None;self.previous_raw_range_cm=None
            self.previous_context=None
            result.stabilization_reset_reason='no_current_path'
            return result
        if not np.isfinite(raw_path).all() or np.any(np.diff(raw_path[:,0])<=0):
            self.previous_path_cm=None;self.previous_status=None;self.previous_raw_range_cm=None
            self.previous_context=None
            raise ValueError('Current raw path must have finite, increasing forward coordinates')
        path=raw_path.copy();previous=self.previous_path_cm
        old_context=self.previous_context or {}
        old_side=old_context.get('reference_side','none')
        new_side=current_context['reference_side']
        side_changed=old_side in ('left','right') and new_side in ('left','right') and old_side!=new_side
        # Two fresh observed boundaries can expose a substantially different
        # shared range. Prefer that new geometry over the old path's filter lag.
        # One-sided ranges already depend on a width estimate, so this reset is
        # deliberately limited to consecutive pairs of observed boundaries.
        paired_range_changed=False
        if (self.pair_range_reset>0 and self.previous_raw_range_cm is not None
                and old_context.get('both_observed',False) and current_context['both_observed']):
            old_near,old_far=self.previous_raw_range_cm
            jump=max(abs(raw_path[0,0]-old_near),abs(raw_path[-1,0]-old_far))
            old_interval=old_context.get('observation_range_cm')
            new_interval=current_context['observation_range_cm']
            if old_interval is not None and new_interval is not None:
                jump=max(jump,abs(new_interval[0]-old_interval[0]),abs(new_interval[1]-old_interval[1]))
            result.stabilization_range_jump_cm=float(jump)
            paired_range_changed=jump>self.pair_range_reset
        if side_changed or paired_range_changed:previous=None
        if previous is None:
            result.stabilization_reset_reason=('observed_side_changed' if side_changed else
                'paired_observation_range_changed' if paired_range_changed else 'first_current_path')
        elif self.tau>0 and self.limit>0:
            lo=max(path[0,0],previous[0,0]);hi=min(path[-1,0],previous[-1,0])
            if hi-lo>=self.min_shared:
                result.shared_range_cm=(float(lo),float(hi))
                inside=(path[:,0]>=lo)&(path[:,0]<=hi)
                x=path[inside,0]
                old_y=np.interp(x,previous[:,0],previous[:,1])
                weight=float(np.exp(-1/(self.fps*self.tau)))
                # A low-degree correction removes interpolation ripple from the
                # previous sampled path. A single scale enforces the bound; hard
                # clipping individual samples would create curvature corners.
                z=(x-lo)/max(hi-lo,1e-9)
                degree=min(2,len(x)-1)
                co=np.polyfit(z,old_y-path[inside,1],degree)
                correction=weight*np.polyval(co,z)
                # The correction reaches zero smoothly at an interior overlap
                # edge. Outside that edge the current raw path is used intact.
                if self.edge>0:
                    taper=np.ones(len(x))
                    for distance,interior in [(x-lo,lo>path[0,0]+1e-9),(hi-x,hi<path[-1,0]-1e-9)]:
                        if interior:
                            t=np.clip(distance/self.edge,0,1)
                            taper*=t*t*(3-2*t)
                    correction*=taper
                maximum=float(np.max(abs(correction)))
                correction*=min(1.,self.limit/max(maximum,1e-12))
                path[inside,1]+=correction
                result.stabilization_previous_weight=weight
                result.max_path_adjustment_cm=float(np.max(abs(path[:,1]-raw_path[:,1])))
                result.stabilization_applied=result.max_path_adjustment_cm>1e-9
            else:result.stabilization_reset_reason='no_shared_current_range'
        else:result.stabilization_reset_reason='disabled'
        if result.stabilization_applied:
            result.path_cm=path
            result.path_uv=self.mapping.cm_to_pixels(path)
            result.curvature_per_cm=sampled_curvature(path)
            # The blended sampled path generally has no single quadratic form.
            result.center_coefficients=None
            result.method=raw.method+'_bounded_temporal'
            result.reason=raw.reason+'; path stabilized only inside current shared range'
        self.previous_path_cm=path.copy()
        self.previous_status=raw.status
        self.previous_raw_range_cm=(float(raw_path[0,0]),float(raw_path[-1,0]))
        self.previous_context=current_context.copy()
        return result
