"""Plan from one centre/width model jointly constrained by real boundary rows."""
import numpy as np
from .metric_planning import MetricPlan,fit_metric_boundary,normal_offset
from .stable_metric_planning import StableMetricLanePlanner
from .yellow_guided_planning import YellowGuidedLanePlanner
from .joint_lane_model import fit_joint_lane


class JointBSplineLanePlanner(StableMetricLanePlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fallback=YellowGuidedLanePlanner(*args,**kwargs)

    def _fallback(self,backup,reason,diagnostics=None):
        self.previous_path_cm=None;self.previous_context=None;self.previous_raw_range_cm=None
        backup.joint_fit_accepted=False;backup.joint_fit_reason=reason
        backup.joint_diagnostics=diagnostics or {}
        backup.joint_model=None;backup.joint_evidence=[]
        backup.status='FALLBACK_'+backup.status
        backup.reason='Joint fit rejected: '+reason+'; '+backup.reason
        return backup

    def plan(self,lanes):
        def fallback(reason,diagnostics=None):
            # A lazily evaluated fallback must not reuse history from an old
            # failure. Fit CURRENT rows with a fresh bounded-filter state.
            self.fallback.previous_path_cm=None;self.fallback.previous_status=None
            self.fallback.previous_context=None;self.fallback.previous_raw_range_cm=None
            return self._fallback(self.fallback.plan(lanes),reason,diagnostics)
        if not lanes.usable:return fallback('no_current_observation')
        fits={};evidence=[];rejected=[]
        for side in ('left','right'):
            fit=fit_metric_boundary(getattr(lanes,side),self.mapping,self.cfg,self.windows,self.height)
            fits[side]=fit
            part=fit_metric_boundary(getattr(lanes,'partial_'+side,None),self.mapping,self.cfg,self.windows,self.height)
            for observed,partial in [(fit,False),(part,True)]:
                if observed is None:continue
                points=observed.row_centers_cm[observed.inlier_rows]
                confidence=1/(1+(observed.rmse_cm/2.)**2)
                # This is an INITIAL guess only; it is never added as observation.
                d=-self.width/2 if side=='left' else self.width/2
                if np.any(1-d*observed.curvature(points[:,0])<=self.cfg['offset_singularity_margin']):
                    if partial:rejected.append(side+': folded partial initial offset');continue
                    return fallback('full_observation_initial_offset_folds')
                evidence.append({'side':side,'partial':partial,'points_cm':points,'confidence':confidence,
                    'center_hint_cm':normal_offset(observed,points[:,0],d)})
        if not evidence:return fallback('no_current_metric_rows')
        def fit(items):
            hint=np.vstack([e['center_hint_cm'] for e in items]);hint=hint[np.isfinite(hint).all(axis=1)]
            lo=max(0.,float(hint[:,0].min()));hi=float(hint[:,0].max())
            if hi-lo<self.cfg['min_path_span_cm']:return None,(lo,hi)
            bins=np.floor((hint[:,0]-lo)/10).astype(int)
            centres=np.asarray([np.median(hint[bins==b],axis=0) for b in np.unique(bins)])
            z=(centres[:,0]-lo)/max(hi-lo,1.)
            co=np.polyfit(z,centres[:,1],min(2,len(centres)-1))
            t=np.linspace(lo,hi,int(self.cfg['samples']))
            initial=np.column_stack((t,np.polyval(co,(t-lo)/(hi-lo))))
            return fit_joint_lane(items,(lo,hi),initial,self.width,self.cfg),(lo,hi)
        model,(lo,hi)=fit(evidence)
        if model is None:return fallback('current_support_too_short')
        bad=[]
        for info in model.evidence_weights:
            if not info['partial']:continue
            keep=np.asarray(info['inlier'],bool)&(np.asarray(info['weights'])>0)
            pp=np.asarray(info['projected_parameter_cm'])[keep]
            if keep.sum()<6 or (len(pp) and np.ptp(pp)<8.):bad.append(info['source_index'])
        if bad:
            rejected.extend(evidence[k]['side']+': insufficient partial joint inliers' for k in bad)
            evidence=[e for k,e in enumerate(evidence) if k not in bad]
            # Remove rejected fragments from the domain and initialisation too.
            model,(lo,hi)=fit(evidence)
            if model is None:return fallback('remaining_support_too_short')
        if not model.accepted:return fallback(model.reason,model.diagnostics)
        t=np.linspace(lo,hi,int(self.cfg['samples']))
        path=model.center(t);lp=model.boundary('left',t);rp=model.boundary('right',t)
        curvature=model.curvature(t);width=model.width(t)
        if (not all(np.isfinite(p).all() for p in [path,lp,rp,curvature,width])
                or np.any(np.diff(lp[:,0])<=0) or np.any(np.diff(rp[:,0])<=0)
                or np.max(abs(curvature))>float(self.cfg.get('joint_max_model_curvature_per_cm',.02))):
            return fallback('full_model_geometry_guard',model.diagnostics)
        actual_sides={e['side'] for e in evidence if not e['partial']}
        partial_sides={e['side'] for e in evidence if e['partial']}
        status=('JOINT_TWO_SIDES' if len(actual_sides)==2 else
                'JOINT_PARTIAL_SUPPORT' if len(actual_sides|partial_sides)==2 else
                'JOINT_LEFT_ONLY' if 'left' in actual_sides else 'JOINT_RIGHT_ONLY')
        raw=MetricPlan(status,'cubic B-spline centre and normal width jointly fitted to CURRENT real rows',
            fits['left'],fits['right'],path,self.mapping.cm_to_pixels(path),None,None,curvature,'joint_bspline_normal_width')
        # No left/right preferred reference; actual two-sided ranges still reset.
        result=self.stabilize(raw,context={'reference_side':'none','both_observed':len(actual_sides)==2,
            'observation_range_cm':(lo,hi)})
        result.joint_fit_accepted=True;result.joint_fit_reason=model.reason
        result.joint_model=model;result.joint_evidence=evidence;result.joint_diagnostics=model.diagnostics
        result.joint_rejected_partials=rejected
        result.joint_domain_cm=(lo,hi)
        result.full_model_left_cm=lp;result.full_model_right_cm=rp
        result.full_model_parameter_cm=t
        result.full_model_left_inferred_mask=np.ones(len(t),bool)
        result.full_model_right_inferred_mask=np.ones(len(t),bool)
        result.full_model_left_observation_weight=np.zeros(len(t))
        result.full_model_reference_side='joint';result.full_model_reason=raw.reason
        result.yellow_guided=False
        result.boundary_pair_left_cm=lp;result.boundary_pair_right_cm=rp
        result.boundary_pair_left_uv=self.mapping.cm_to_pixels(lp);result.boundary_pair_right_uv=self.mapping.cm_to_pixels(rp)
        result.boundary_pair_midpoint_cm=(lp+rp)/2
        result.boundary_pair_midpoint_error_cm=float(np.max(np.linalg.norm((lp+rp)/2-path,axis=1)))
        result.boundary_pair_display_path_offset_cm=float(np.max(np.linalg.norm(path-result.path_cm,axis=1)))
        result.boundary_pair_left_estimated=True;result.boundary_pair_right_estimated=True
        result.boundary_pair_available=True;result.boundary_pair_provenance='joint_model_from_real_rows'
        result.boundary_pair_reason='left/right generated from fitted centre and normal half-width; not independent observations'
        result.blue_estimated_fraction=1.
        result.width_model_cm=float(model.width(np.array([(lo+hi)/2]))[0])
        result.width_model_min_cm=float(width.min());result.width_model_max_cm=float(width.max())
        result.width_model_type='joint';result.width_evidence_kind='joint_real_rows'
        result.width_model_slope=float(model.width_slope_cm_per_cm)
        result.width_evidence_reason=raw.reason
        result.partial_used=bool(partial_sides)
        # A support envelope is not a synthetic measured point; gaps stay labelled.
        support={'left':[],'right':[]};all_parameters=[]
        for info in model.evidence_weights:
            keep=np.asarray(info['inlier'],bool)&(np.asarray(info['weights'])>0)
            s=np.asarray(info['projected_parameter_cm'])[keep]
            support[info['side']].extend(s.tolist());all_parameters.extend(s.tolist())
        support={side:np.asarray(s) for side,s in support.items()}
        near={side:(np.min(abs(t[:,None]-p[None,:]),axis=1)<=2. if len(p) else np.zeros(len(t),bool)) for side,p in support.items()}
        result.full_model_left_observation_weight=near['left'].astype(float)
        result.joint_support_kind=np.full(len(t),'gap',dtype=object)
        result.joint_support_kind[near['left']]='left_only';result.joint_support_kind[near['right']]='right_only'
        result.joint_support_kind[near['left']&near['right']]='two_sides'
        if all_parameters:
            result.joint_support_kind[(t<min(all_parameters))|(t>max(all_parameters))]='extrapolated'
        return result
