"""Complete the blue boundary over the CURRENT yellow boundary's full support.

The left observation interval never sets the yellow-guided model's range. A
current yellow tangent and a robust constant normal width predict missing blue
segments. Current left rows constrain the scalar width where they actually
exist and agree with a plausible opposite boundary. No old blue curve
or invented observation pixels are needed for this completion.
"""
import numpy as np
from .metric_planning import MetricPlan,MetricLanePlanner,fit_metric_boundary,normal_offset
from .stable_metric_planning import StableMetricLanePlanner,sampled_curvature
from .boundary_pair_planning import attach_boundary_pair
from .normal_width_model import estimate_normal_width


class YellowGuidedLanePlanner(StableMetricLanePlanner):
    """Continue a regularized normal-width model from valid CURRENT yellow data."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.width_good=float(self.cfg.get('blue_width_good_relative_error',.10))
        self.width_bad=float(self.cfg.get('blue_width_max_relative_error',.30))
        self.angle_good=float(self.cfg.get('blue_angle_good_rad',.15))
        self.angle_bad=float(self.cfg.get('blue_angle_max_rad',.45))
        if (not np.isfinite([self.width_good,self.width_bad,self.angle_good,self.angle_bad]).all()
                or not 0<=self.width_good<self.width_bad
                or not 0<=self.angle_good<self.angle_bad):
            raise ValueError('Valid blue normal-width geometry confidence settings required')
        for key,default in [('blue_width_prior_support_cm',30.),('blue_width_max_adjustment_cm',12.),
                            ('blue_partial_weight',.25),('blue_partial_max_adjustment_cm',1.5),
                            ('blue_width_min_support_cm',8.),('blue_width_max_scatter_cm',8.),
                            ('blue_partial_max_scatter_cm',3.),('blue_partial_max_support_cm',30.)]:
            value=float(self.cfg.get(key,default))
            if not np.isfinite(value) or value<=0:raise ValueError(f'Positive finite {key} required')

    def _metadata(self,plan,left_pair,right_pair,parameters,inferred_mask,group,weights):
        """Keep complete display models separate from the forward-valid path pair."""
        plan.full_model_left_cm=np.asarray(left_pair).copy()
        plan.full_model_right_cm=np.asarray(right_pair).copy()
        plan.full_model_parameter_cm=np.asarray(parameters).copy()
        plan.full_model_left_inferred_mask=np.asarray(inferred_mask,dtype=bool).copy()
        plan.full_model_right_inferred_mask=np.zeros(len(parameters),bool)
        plan.full_model_left_observation_weight=np.asarray(weights).copy()
        plan.full_model_reference_side='right'
        plan.full_model_reason='current yellow tangent and a robust low-degree normal-width model across full support'
        plan.yellow_guided=True
        plan.yellow_observed_interval_cm=(float(parameters[0]),float(parameters[-1]))
        plan.blue_observation_interval_cm=plan.left.interval_cm if plan.left is not None else None
        plan.blue_estimated_fraction=float(np.mean(inferred_mask))
        plan.boundary_pair_left_cm=left_pair[group].copy()
        plan.boundary_pair_right_cm=right_pair[group].copy()
        plan.boundary_pair_left_uv=self.mapping.cm_to_pixels(plan.boundary_pair_left_cm)
        plan.boundary_pair_right_uv=self.mapping.cm_to_pixels(plan.boundary_pair_right_cm)
        plan.boundary_pair_left_estimated=bool(np.any(inferred_mask[group]))
        plan.boundary_pair_right_estimated=False
        midpoint=(plan.boundary_pair_left_cm+plan.boundary_pair_right_cm)/2
        plan.boundary_pair_midpoint_cm=midpoint
        plan.boundary_pair_midpoint_error_cm=float(np.max(np.linalg.norm(midpoint-plan.raw_path_cm,axis=1)))
        plan.boundary_pair_display_path_offset_cm=float(np.max(np.linalg.norm(midpoint-plan.path_cm,axis=1)))
        plan.boundary_pair_available=True
        plan.boundary_pair_provenance='yellow_guided_global_normal_width_with_current_left_support' if np.any(weights>0) else 'yellow_guided_inferred_blue'
        plan.boundary_pair_reason='paired normal parameters form the raw center; blue coverage follows current yellow support'
        return plan

    def _fallback(self,lanes):
        plan=self.stabilize(MetricLanePlanner.plan(self,lanes))
        attach_boundary_pair(plan,self.mapping,self.cfg,self.width)
        empty=np.empty((0,2))
        plan.full_model_left_cm=plan.boundary_pair_left_cm.copy()
        plan.full_model_right_cm=plan.boundary_pair_right_cm.copy()
        plan.full_model_parameter_cm=(plan.full_model_left_cm[:,0].copy() if len(plan.full_model_left_cm) else np.array([]))
        plan.full_model_left_inferred_mask=np.full(len(plan.full_model_left_cm),plan.left is None,bool)
        plan.full_model_right_inferred_mask=np.full(len(plan.full_model_right_cm),plan.right is None,bool)
        plan.full_model_left_observation_weight=np.ones(len(plan.full_model_left_cm)) if plan.left is not None else np.zeros(len(plan.full_model_left_cm))
        plan.full_model_reference_side='left' if plan.left is not None else 'none'
        plan.full_model_reason='current yellow unavailable; current-left fallback only'
        plan.yellow_guided=False;plan.yellow_observed_interval_cm=None
        plan.blue_observation_interval_cm=plan.left.interval_cm if plan.left is not None else None
        plan.blue_estimated_fraction=1. if plan.left is None else 0.
        return plan

    def _invalid(self,status,reason,left,right):
        empty=np.empty((0,2))
        raw=MetricPlan(status,reason,left,right,empty,empty,None,None,np.array([]),'none')
        plan=self.stabilize(raw)
        attach_boundary_pair(plan,self.mapping,self.cfg,self.width)
        plan.full_model_left_cm=empty.copy();plan.full_model_right_cm=empty.copy()
        plan.full_model_parameter_cm=np.array([])
        plan.full_model_left_inferred_mask=np.array([],bool);plan.full_model_right_inferred_mask=np.array([],bool)
        plan.full_model_left_observation_weight=np.array([])
        plan.full_model_reference_side='right';plan.full_model_reason=reason;plan.yellow_guided=False
        plan.yellow_observed_interval_cm=right.interval_cm if right is not None else None
        plan.blue_observation_interval_cm=left.interval_cm if left is not None else None
        plan.blue_estimated_fraction=0.
        return plan

    def plan(self,lanes):
        if not lanes.usable:return self._fallback(lanes)
        right=fit_metric_boundary(lanes.right,self.mapping,self.cfg,self.windows,self.height)
        if right is None:return self._fallback(lanes)
        left=fit_metric_boundary(lanes.left,self.mapping,self.cfg,self.windows,self.height)
        lo,hi=right.interval_cm
        if hi-lo<self.cfg['min_path_span_cm']:
            return self._invalid('TOO_SHORT','current yellow interval too short',left,right)
        t=np.linspace(lo,hi,self.cfg['samples'])
        rp=np.column_stack((t,right.y(t)))
        partial=None
        if left is None and self.cfg.get('blue_use_partial',True):
            partial=fit_metric_boundary(getattr(lanes,'partial_left',None),self.mapping,self.cfg,self.windows,self.height)
        estimate=estimate_normal_width(right,left if left is not None else partial,self.width,self.cfg,partial=left is None)
        width_values=estimate.evaluate(t)
        width=estimate.width_cm;model_type=estimate.model
        nominal=normal_offset(right,t,width_values)
        regular=(1-width_values*right.curvature(t)>self.cfg['offset_singularity_margin'])
        fallback_reason=''
        if model_type=='affine':
            constant=normal_offset(right,t,estimate.constant_width_cm)
            center_candidate=(nominal+rp)/2;center_constant=(constant+rp)/2
            slope=right.slope(t)
            blue_dx_dt=1-width_values*right.curvature(t)-estimate.slope_cm_per_cm*slope/np.sqrt(1+slope*slope)
            geometry_ok=(np.all(regular) and np.all(np.diff(nominal[:,0])>0)
                and np.all(blue_dx_dt>0) and np.all(np.diff(center_candidate[:,0])>0))
            if geometry_ok:
                # Shape guard relative to the current constant-width candidate;
                # this is a model regularizer, not a vehicle turning limit.
                baseline_peak=float(np.max(abs(sampled_curvature(center_constant))))
                candidate_peak=float(np.max(abs(sampled_curvature(center_candidate))))
                geometry_ok=candidate_peak<=1.5*baseline_peak+.0005
            if not geometry_ok:
                fallback_reason='affine_geometry_guard; reverted_to_current_constant_fit'
                width=estimate.constant_width_cm;width_values=np.full(len(t),width)
                nominal=constant;model_type='constant'
                regular=(1-width_values*right.curvature(t)>self.cfg['offset_singularity_margin'])
        if (not np.all(regular) or np.any(np.diff(nominal[:,0])<=0)) and width!=self.width:
            fallback_reason='fitted_width_offset_invalid; reverted_to_nominal_width'
            width=self.width;nominal=normal_offset(right,t,width)
            width_values=np.full(len(t),width);model_type='nominal'
            regular=(1-width*right.curvature(t)>self.cfg['offset_singularity_margin'])
        if (not np.isfinite(nominal).all() or not np.all(regular)
                or np.any(np.diff(nominal[:,0])<=0)):
            return self._invalid('INVALID_BLUE_OFFSET','current yellow normal offset folds; no valid full blue model',left,right)
        # These weights describe real support only; they do NOT blend coordinates.
        weights=np.zeros(len(t))
        if estimate.accepted_rows and model_type!='nominal':
            weights=np.interp(t,estimate.parameter_cm,estimate.weights,left=0.,right=0.)
            # Do not label the empty gap between separated observed dashes as
            # direct row support. These diagnostic weights never shape a curve.
            ids=np.searchsorted(estimate.parameter_cm,t)
            a=estimate.parameter_cm[np.clip(ids-1,0,len(estimate.parameter_cm)-1)]
            b=estimate.parameter_cm[np.clip(ids,0,len(estimate.parameter_cm)-1)]
            weights[np.minimum(abs(t-a),abs(t-b))>1.]=0.
        lp=nominal
        if not np.isfinite(lp).all() or np.any(lp[:,1]<=rp[:,1]):
            return self._invalid('CROSSED_BLUE_MODEL','current yellow-guided blue model crosses its right partner',left,right)
        center=(lp+rp)/2
        valid=np.isfinite(center).all(axis=1)&(center[:,0]>=0)
        ids=np.flatnonzero(valid)
        groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
        groups=[g for g in groups if len(g)>=3 and np.ptp(center[g,0])>=self.cfg['min_path_span_cm']]
        if not groups:return self._invalid('NO_FORWARD_CENTER','yellow-guided midpoint has insufficient forward range',left,right)
        group=max(groups,key=lambda g:np.ptp(center[g,0]));path=center[group]
        if np.any(np.diff(path[:,0])<=0):return self._invalid('FOLDED_CENTER','yellow-guided midpoint folds',left,right)
        # A width-constrained blue curve remains a model even where data support it.
        inferred=np.ones(len(t),bool)
        status='ESTIMATED_FROM_RIGHT_WITH_LEFT_SUPPORT' if np.any(weights>0) else 'ESTIMATED_FROM_RIGHT'
        if len(group)<len(center):status+='_TRUNCATED'
        raw=MetricPlan(status,'full current yellow support; robust low-degree normal width constrained by current real rows',
            left,right,path,self.mapping.cm_to_pixels(path),None,lp[group],sampled_curvature(path),'yellow_guided_global_normal_width')
        overlap=(max(left.interval_cm[0],lo),min(left.interval_cm[1],hi)) if left is not None else None
        if overlap is not None and overlap[1]<=overlap[0]:overlap=None
        plan=self.stabilize(raw,context={'reference_side':'right','both_observed':left is not None,
            'observation_range_cm':overlap})
        plan.width_model_cm=float(width);plan.width_model_type=model_type
        plan.width_model_slope=float(estimate.slope_cm_per_cm) if model_type=='affine' else 0.
        plan.width_model_min_cm=float(np.min(width_values));plan.width_model_max_cm=float(np.max(width_values))
        plan.width_evidence_kind=estimate.kind if model_type!='nominal' else 'none'
        plan.width_evidence_reason=fallback_reason or estimate.reason
        plan.width_accepted_rows=estimate.accepted_rows
        plan.width_support_span_cm=estimate.support_span_cm;plan.width_scatter_cm=estimate.scatter_cm
        plan.partial_used=estimate.kind=='partial' and model_type!='nominal'
        plan.partial_width_adjustment_cm=float(width-self.width) if plan.partial_used else 0.
        return self._metadata(plan,lp,rp,t,inferred,group,weights)
