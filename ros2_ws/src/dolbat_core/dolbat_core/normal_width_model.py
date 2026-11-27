"""Robust low-degree normal distance from current, real row-centre observations.

A constant model is the default. Optional affine variation must improve a
well-supported fit and obey global width and width-gradient limits. Neither
model introduces observation-interval blend edges into the geometry.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class NormalWidthEstimate:
    width_cm: float
    kind: str
    reason: str
    parameter_cm: np.ndarray
    weights: np.ndarray
    distances_cm: np.ndarray
    accepted_rows: int = 0
    support_span_cm: float = 0.
    scatter_cm: float = 0.
    slope_cm_per_cm: float = 0.
    model: str = 'constant'
    reference_midpoint_cm: float = 0.
    reference_scale_cm: float = 1.
    constant_width_cm: float | None = None
    model_rmse_cm: float = 0.
    constant_rmse_cm: float = 0.

    def evaluate(self,parameter_cm):
        """Normal distance on the CURRENT reference parameter, in cm."""
        parameter=np.asarray(parameter_cm,dtype=float)
        return self.width_cm+self.slope_cm_per_cm*(parameter-self.reference_midpoint_cm)


def _bounded_affine_fit(t,d,base,reference_interval,nominal,constant_width,config):
    """Fit d=nominal+a+b*z with ridge priors on a and b, using real rows.

    Weight normalization gives the data term units of supported cm, matching
    the nominal-width and zero-slope prior strengths. A single coefficient
    scale enforces limits over the complete reference, without clipping points.
    """
    lo,hi=map(float,reference_interval);mid=(lo+hi)/2;scale=(hi-lo)/2
    span=float(np.ptp(t))
    minimum_span=float(config.get('blue_width_affine_min_support_cm',80.))
    minimum_fraction=float(config.get('blue_width_affine_min_support_fraction',.35))
    if scale<=0 or span<minimum_span or span<minimum_fraction*(hi-lo):return None
    z=(t-mid)/scale;design=np.column_stack((np.ones(len(t)),z))
    # A nearly singular design cannot support an independent width gradient.
    if not np.isfinite(design).all() or np.linalg.matrix_rank(design)<2:return None
    prior=float(config.get('blue_width_prior_support_cm',30.))
    slope_prior=float(config.get('blue_width_affine_prior_support_cm',30.))
    cap=float(config.get('blue_width_max_adjustment_cm',12.))
    gradient_limit=float(config.get('blue_width_affine_max_gradient',.10))
    relative_improvement=float(config.get('blue_width_affine_min_relative_improvement',.20))
    absolute_improvement=float(config.get('blue_width_affine_min_absolute_improvement_cm',.30))
    settings=np.array([minimum_span,minimum_fraction,prior,slope_prior,cap,gradient_limit,
        relative_improvement,absolute_improvement])
    if (not np.isfinite(settings).all() or min(prior,slope_prior,cap,gradient_limit)<=0
            or minimum_span<0 or not 0<=minimum_fraction<=1
            or not 0<=relative_improvement<1 or absolute_improvement<0):
        raise ValueError('Finite, positive affine normal-width priors and valid acceptance settings required')
    theta=np.array([constant_width-nominal,0.]);weights=base.copy()
    for _ in range(6):
        residual=d-(nominal+design@theta)
        robust_scale=max(1.,1.4826*float(np.median(abs(residual-np.median(residual)))))
        weights=base*np.minimum(1.,2.5*robust_scale/np.maximum(abs(residual),1e-12))
        strength=span*float(np.mean(weights))
        precision=strength*weights/float(np.sum(weights))
        matrix=design.T@(precision[:,None]*design)+np.diag([prior,slope_prior])
        rhs=design.T@(precision*(d-nominal))
        theta=np.linalg.solve(matrix,rhs)
    maximum_delta=float(max(abs(theta[0]-theta[1]),abs(theta[0]+theta[1])))
    maximum_gradient=abs(theta[1]/scale)
    coefficient_scale=min(1.,cap/max(maximum_delta,1e-12),gradient_limit/max(maximum_gradient,1e-12))
    theta*=coefficient_scale
    fitted=nominal+design@theta
    # Require a regular normal correspondence for every accepted actual row.
    # The caller additionally checks full-model geometry on the whole reference.
    if not np.isfinite(fitted).all() or np.any(fitted<=0):return None
    constant_rmse=float(np.sqrt(np.average((d-constant_width)**2,weights=weights)))
    fitted_rmse=float(np.sqrt(np.average((d-fitted)**2,weights=weights)))
    improvement=constant_rmse-fitted_rmse
    if (abs(theta[1]/scale)<1e-10 or improvement<absolute_improvement
            or fitted_rmse>(1-relative_improvement)*constant_rmse):return None
    return float(nominal+theta[0]),float(theta[1]/scale),weights,fitted_rmse,constant_rmse


def estimate_normal_width(reference, observation, nominal_cm, config, partial=False):
    """Project real inlier row centres onto the CURRENT reference quadratic.

    A Newton closest-point solve supplies signed normal distances. Endpoint
    clamping is not evidence: tangential residual and offset regularity must
    also pass. A nominal-width prior shrinks short or weak support; partial
    fragments have their own low weight and bounded total influence.
    """
    empty=np.array([])
    lo,hi=map(float,reference.interval_cm)
    reference_midpoint=(lo+hi)/2
    reference_scale=max((hi-lo)/2,1e-9)
    def no(reason):
        return NormalWidthEstimate(float(nominal_cm),'none',reason,empty,empty,empty,
            reference_midpoint_cm=reference_midpoint,reference_scale_cm=reference_scale,
            constant_width_cm=float(nominal_cm))
    if observation is None:return no('no_current_left_observation')
    points=np.asarray(observation.row_centers_cm)[observation.inlier_rows]
    if len(points)<12:return no('too_few_current_rows')
    x,y=points.T
    t=np.clip(x,lo,hi)
    second=float(np.polyval(np.polyder(reference.coefficients,2),0.))
    for _ in range(20):
        f=reference.y(t);slope=reference.slope(t)
        numerator=t-x+(f-y)*slope
        denominator=1+slope*slope+(f-y)*second
        step=np.divide(numerator,denominator,out=np.zeros_like(t),where=abs(denominator)>1e-8)
        t=np.clip(t-np.clip(step,-20.,20.),lo,hi)
    f=reference.y(t);slope=reference.slope(t);q=np.sqrt(1+slope*slope)
    distance=(-(x-t)*slope+(y-f))/q
    tangent=((x-t)+(y-f)*slope)/q
    width_error=abs(distance/nominal_cm-1.)
    angle=abs(np.arctan(observation.slope(x))-np.arctan(slope))
    good=float(config.get('blue_width_good_relative_error',.10))
    bad=float(config.get('blue_width_max_relative_error',.30))
    ag=float(config.get('blue_angle_good_rad',.15));ab=float(config.get('blue_angle_max_rad',.45))
    wc=np.clip((bad-width_error)/(bad-good),0,1)
    ac=np.clip((ab-angle)/(ab-ag),0,1)
    weights=wc*ac
    valid=(np.isfinite(distance)&(abs(tangent)<.5)&(distance>0)&(weights>0)
           &(1-distance*reference.curvature(t)>config['offset_singularity_margin']))
    if valid.sum()<12:return no('width_direction_or_normal_correspondence_rejected')
    t,d,w=t[valid],distance[valid],weights[valid]
    span=float(np.ptp(t))
    if span<float(config.get('blue_width_min_support_cm',8.)):
        return no('normal_support_too_short')
    # Huber reweighting limits isolated row errors without replacing real rows.
    location=float(np.median(d));base=w.copy()
    for _ in range(4):
        residual=d-location
        scale=max(1.,1.4826*float(np.median(abs(residual-np.median(residual)))))
        w=base*np.minimum(1.,2.5*scale/np.maximum(abs(residual),1e-12))
        location=float(np.average(d,weights=w))
    scatter=float(np.sqrt(np.average((d-location)**2,weights=w)))
    maximum_scatter=float(config.get('blue_partial_max_scatter_cm',3.) if partial else config.get('blue_width_max_scatter_cm',8.))
    if scatter>maximum_scatter:return no('normal_distance_scatter_rejected')
    evidence_strength=span*float(np.mean(w))
    prior=float(config.get('blue_width_prior_support_cm',30.))
    cap=float(config.get('blue_width_max_adjustment_cm',12.))
    if partial:
        evidence_strength=min(evidence_strength,float(config.get('blue_partial_max_support_cm',30.)))
        evidence_strength*=float(config.get('blue_partial_weight',.25))
        cap=float(config.get('blue_partial_max_adjustment_cm',1.5))
    adjustment=(location-nominal_cm)*evidence_strength/(prior+evidence_strength)
    adjustment=float(np.clip(adjustment,-cap,cap))
    kind='partial' if partial else 'full'
    width=float(nominal_cm+adjustment);constant_width=width
    model='constant';gradient=0.
    constant_rmse=float(np.sqrt(np.average((d-width)**2,weights=w)))
    model_rmse=constant_rmse
    reason='robust current normal-distance fit with nominal-width prior'
    if not partial and config.get('blue_width_allow_affine',False):
        affine=_bounded_affine_fit(t,d,base,reference.interval_cm,nominal_cm,constant_width,config)
        if affine is not None:
            candidate_width,gradient,candidate_weights,model_rmse,constant_rmse=affine
            candidate=candidate_width+gradient*(t-reference_midpoint)
            regular=1-candidate*reference.curvature(t)>config['offset_singularity_margin']
            if np.all(regular):
                width=candidate_width;w=candidate_weights;model='affine'
                reason='regularized affine normal distance; broad current support and measured residual improvement'
            else:
                gradient=0.;model_rmse=float(np.sqrt(np.average((d-width)**2,weights=w)))
    order=np.argsort(t)
    return NormalWidthEstimate(width,kind,reason,t[order],w[order],d[order],len(t),span,scatter,
        slope_cm_per_cm=gradient,model=model,reference_midpoint_cm=reference_midpoint,
        reference_scale_cm=reference_scale,constant_width_cm=constant_width,
        model_rmse_cm=model_rmse,constant_rmse_cm=constant_rmse)
