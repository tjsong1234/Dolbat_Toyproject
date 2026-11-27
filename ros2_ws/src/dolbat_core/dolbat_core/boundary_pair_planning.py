"""Make the observed/estimated boundary pair behind a continuous center explicit.

This does not promote an estimated boundary to an observed MetricBoundary or add
scatter points. Corresponding points on a normal-offset pair generally have
different forward X coordinates; their VECTOR midpoint is the correct center.
"""
import numpy as np


def attach_boundary_pair(plan,mapping=None,config=None,lane_center_width_cm=None):
    """Attach a current raw boundary pair and return ``plan``.

    ``boundary_pair_midpoint_cm`` is the exact unfiltered center from the current
    boundaries. ``plan.path_cm`` remains the independently bounded temporal path;
    these are kept separate rather than disguising smoothing as an observation.

    For a single observation the existing planner's valid normal-offset partner
    already corresponds to ``raw_path_cm`` (or ``path_cm`` before stabilization).
    The observed points are recovered as ``2*center - partner`` and checked
    against the current observed function. Thus clipped/truncated parameter
    ranges remain exact, without solving an inverse curve or extrapolating it.
    """
    empty=np.empty((0,2),dtype=float)
    plan.boundary_pair_left_cm=empty.copy()
    plan.boundary_pair_right_cm=empty.copy()
    plan.boundary_pair_left_uv=empty.copy()
    plan.boundary_pair_right_uv=empty.copy()
    plan.boundary_pair_midpoint_cm=empty.copy()
    plan.boundary_pair_available=False
    plan.boundary_pair_left_estimated=plan.left is None
    plan.boundary_pair_right_estimated=plan.right is None
    plan.boundary_pair_provenance='unavailable'
    plan.boundary_pair_reason='no current usable path'
    plan.boundary_pair_midpoint_error_cm=None
    plan.boundary_pair_display_path_offset_cm=None
    if not plan.usable:return plan

    raw=np.asarray(getattr(plan,'raw_path_cm',plan.path_cm),dtype=float)
    if raw.ndim!=2 or raw.shape[1]!=2 or len(raw)<2 or not np.isfinite(raw).all():
        plan.boundary_pair_reason='current raw path is invalid'
        return plan
    if np.any(np.diff(raw[:,0])<=0):
        plan.boundary_pair_reason='current raw path has no increasing forward range'
        return plan
    margin=float((config or {}).get('offset_singularity_margin',.05))
    tolerance=1e-6
    left=plan.left;right=plan.right
    if left is not None and right is not None:
        x=raw[:,0]
        if (x[0]<max(left.interval_cm[0],right.interval_cm[0])-tolerance
                or x[-1]>min(left.interval_cm[1],right.interval_cm[1])+tolerance):
            plan.boundary_pair_reason='raw path exceeds the shared observed interval'
            return plan
        lp=np.column_stack((x,left.y(x)));rp=np.column_stack((x,right.y(x)))
        if np.any(lp[:,1]<=rp[:,1]):
            plan.boundary_pair_reason='current observed boundaries cross'
            return plan
        provenance='two_observed_functions'
    elif (left is None)!=(right is None):
        boundary=left if left is not None else right
        partner=plan.inferred_boundary_cm
        if partner is None:
            plan.boundary_pair_reason='normal-offset partner unavailable or folded'
            return plan
        partner=np.asarray(partner,dtype=float)
        if partner.shape!=raw.shape or not np.isfinite(partner).all():
            plan.boundary_pair_reason='normal-offset partner does not match the current raw range'
            return plan
        observed=2*raw-partner
        x=observed[:,0]
        if (x.min()<boundary.interval_cm[0]-tolerance
                or x.max()>boundary.interval_cm[1]+tolerance
                or np.any(np.diff(x)<=0)
                or np.max(abs(observed[:,1]-boundary.y(x)))>tolerance):
            plan.boundary_pair_reason='partner is not paired with the current observed function'
            return plan
        distance=np.linalg.norm(partner-observed,axis=1)
        width=float(np.median(distance)) if lane_center_width_cm is None else float(lane_center_width_cm)
        if not np.isfinite(width) or width<=0:raise ValueError('Positive finite lane-center width required')
        if np.max(abs(distance-width))>tolerance:
            plan.boundary_pair_reason='partner distance disagrees with the configured lane width'
            return plan
        signed_width=-width if left is not None else width
        if np.any(1-signed_width*boundary.curvature(x)<=margin):
            plan.boundary_pair_reason='normal-offset partner folds or reaches a singularity'
            return plan
        if left is not None:lp,rp=observed,partner
        else:lp,rp=partner,observed
        provenance='observed_left_plus_estimated_right' if left is not None else 'estimated_left_plus_observed_right'
    else:
        plan.boundary_pair_reason='no current observed boundary'
        return plan

    if not np.isfinite(lp).all() or not np.isfinite(rp).all():
        plan.boundary_pair_reason='nonfinite boundary pair'
        return plan
    midpoint=(lp+rp)/2
    error=float(np.max(np.linalg.norm(midpoint-raw,axis=1)))
    if error>tolerance:
        plan.boundary_pair_reason='boundary midpoint does not reproduce the current raw center'
        return plan
    plan.boundary_pair_left_cm=lp.copy();plan.boundary_pair_right_cm=rp.copy()
    plan.boundary_pair_midpoint_cm=midpoint
    plan.boundary_pair_midpoint_error_cm=error
    plan.boundary_pair_provenance=provenance
    plan.boundary_pair_reason='current observed/estimated functions form a valid midpoint pair'
    plan.boundary_pair_available=True
    if mapping is not None:
        plan.boundary_pair_left_uv=mapping.cm_to_pixels(lp)
        plan.boundary_pair_right_uv=mapping.cm_to_pixels(rp)
    final=np.asarray(plan.path_cm,dtype=float)
    if final.shape==midpoint.shape:
        plan.boundary_pair_display_path_offset_cm=float(np.max(np.linalg.norm(final-midpoint,axis=1)))
    return plan
