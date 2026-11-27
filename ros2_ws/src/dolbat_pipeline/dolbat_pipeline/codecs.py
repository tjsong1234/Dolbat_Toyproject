"""Typed ROS message adapters; no pickle, object arrays or synthetic evidence."""
from types import SimpleNamespace
import json
import numpy as np
from dolbat_interfaces.msg import LaneCurve,LaneFrame,PathFrame,ControlFrame
from dolbat_core.lane_detection import LaneFit,LaneResult
from dolbat_core.metric_planning import MetricBoundary,MetricPlan
from dolbat_core.final_visualization import boundary_curves
from dolbat_core.week4_control import PurePursuitResult
from .common import copy_frame_fields


def points(values):
    result=np.asarray(values,dtype=np.float64)
    if result.size%2:raise ValueError('Coordinate array must contain complete XY pairs')
    result=result.reshape(-1,2)
    if not np.isfinite(result).all():raise ValueError('Coordinate array contains nonfinite values')
    return result


def flattened(values):
    return points(values).reshape(-1).tolist()


def _json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Unsupported metadata type {type(value).__name__}')


def encode_curve(fit):
    message=LaneCurve();message.valid=fit is not None
    if fit is None:return message
    message.coefficients=np.asarray(fit.coefficients,dtype=float).tolist()
    message.pixels_uv=flattened(fit.pixels)
    for name in ('residual_px','coverage','y_min','y_max'):setattr(message,name,float(getattr(fit,name)))
    message.inferred=bool(fit.inferred)
    message.support_json=json.dumps(getattr(fit,'support',{}),default=_json_default,allow_nan=False,separators=(',',':'))
    return message


def decode_curve(message):
    if not message.valid:return None
    co=np.asarray(message.coefficients,dtype=float)
    if co.shape!=(3,) or not np.isfinite(co).all():raise ValueError('Invalid lane coefficients')
    fit=LaneFit(co,points(message.pixels_uv),float(message.residual_px),float(message.coverage),
                float(message.y_min),float(message.y_max),bool(message.inferred))
    fit.support=json.loads(message.support_json or '{}')
    if not isinstance(fit.support,dict):raise ValueError('Lane support metadata must be an object')
    return fit


def encode_lanes(lanes,source):
    message=copy_frame_fields(source,LaneFrame())
    message.status=str(lanes.status);message.reason=str(lanes.reason);message.confidence=float(lanes.confidence)
    for name in ('left','right','partial_left','partial_right'):
        setattr(message,name,encode_curve(getattr(lanes,name,None)))
    message.partial_summary_json=json.dumps(getattr(lanes,'partial_summary',{}),default=_json_default,allow_nan=False,separators=(',',':'))
    message.context_rejections_json=json.dumps(getattr(lanes,'context_rejections',[]),default=_json_default,allow_nan=False,separators=(',',':'))
    return message


def decode_lanes(message):
    result=LaneResult(decode_curve(message.left),decode_curve(message.right),message.status,message.reason,
                      float(message.confidence),np.empty((0,0,3),np.uint8),np.empty(0),[])
    result.partial_left=decode_curve(message.partial_left);result.partial_right=decode_curve(message.partial_right)
    result.partial_summary=json.loads(message.partial_summary_json or '{}')
    result.context_rejections=json.loads(message.context_rejections_json or '[]')
    return result


def encode_plan(plan,source):
    message=copy_frame_fields(source,PathFrame())
    message.status=str(plan.status);message.reason=str(plan.reason);message.method=str(plan.method)
    message.perception_status=str(source.status)
    message.path_bev_cm=flattened(plan.path_cm)
    curves={side:p for side,p,color,inferred in boundary_curves(plan)}
    message.left_model_bev_cm=flattened(curves.get('left',np.empty((0,2))))
    message.right_model_bev_cm=flattened(curves.get('right',np.empty((0,2))))
    for side in ('left','right'):
        boundary=getattr(plan,side,None)
        setattr(message,side+'_observed',boundary is not None)
        if boundary is not None:
            setattr(message,side+'_coefficients',np.asarray(boundary.coefficients,float).tolist())
            setattr(message,side+'_observed_rows_cm',flattened(boundary.row_centers_cm))
        partial=getattr(plan,'partial_points_cm',{}).get(side,np.empty((0,2)))
        setattr(message,'partial_'+side+'_cm',flattened(partial))
    message.joint_fit_accepted=bool(getattr(plan,'joint_fit_accepted',False))
    message.max_path_adjustment_cm=float(getattr(plan,'max_path_adjustment_cm',0.))
    return message


def _observed_boundary(message,side):
    if not getattr(message,side+'_observed'):return None
    rows=points(getattr(message,side+'_observed_rows_cm'))
    coefficients=np.asarray(getattr(message,side+'_coefficients'),float)
    if not np.isfinite(coefficients).all():raise ValueError('Invalid observed boundary coefficients')
    interval=(float(rows[:,0].min()),float(rows[:,0].max())) if len(rows) else (0.,0.)
    return MetricBoundary(coefficients,rows,rows,np.ones(len(rows),bool),interval,0.,0.)


def decode_plan(message):
    path=points(message.path_bev_cm)
    plan=MetricPlan(message.status,message.reason,_observed_boundary(message,'left'),_observed_boundary(message,'right'),
                    path,np.empty((0,2)),None,None,np.zeros(len(path)),message.method)
    left=points(message.left_model_bev_cm);right=points(message.right_model_bev_cm)
    plan.full_model_left_cm=left;plan.full_model_right_cm=right
    plan.full_model_left_inferred_mask=np.ones(len(left),bool)
    plan.full_model_right_inferred_mask=np.ones(len(right),bool)
    plan.boundary_pair_left_cm=left;plan.boundary_pair_right_cm=right
    plan.partial_points_cm={}
    for side in ('left','right'):
        part=points(getattr(message,'partial_'+side+'_cm'))
        if len(part):plan.partial_points_cm[side]=part
    plan.joint_fit_accepted=bool(message.joint_fit_accepted)
    plan.max_path_adjustment_cm=float(message.max_path_adjustment_cm)
    return plan


def encode_control(result,source):
    message=copy_frame_fields(source,ControlFrame())
    message.valid=bool(result.valid);message.status=str(result.status)
    message.candidate_count=int(result.candidate_count)
    message.segment_index=int(result.segment_index) if result.segment_index is not None else -1
    if result.valid:
        message.target_x_m=float(result.target_m[0]);message.target_y_m=float(result.target_m[1])
        for name in ('steering_rad','steering_deg','curvature_per_m','heading_error_rad','segment_fraction','circle_residual_m','actual_lookahead_m'):
            setattr(message,name,float(getattr(result,name)))
    return message


def decode_control(message):
    if not message.valid:
        return PurePursuitResult(False,message.status,candidate_count=int(message.candidate_count))
    return PurePursuitResult(True,message.status,(float(message.target_x_m),float(message.target_y_m)),
        float(message.curvature_per_m),float(message.steering_rad),float(message.steering_deg),
        float(message.heading_error_rad),int(message.segment_index),float(message.segment_fraction),
        float(message.circle_residual_m),int(message.candidate_count),float(message.actual_lookahead_m))
