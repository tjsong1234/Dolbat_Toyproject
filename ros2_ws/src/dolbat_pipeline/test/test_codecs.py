"""Run with generated ROS2 interfaces; no mocked ROS transport claims."""
import json
import numpy as np
import pytest
pytest.importorskip('rclpy')
pytest.importorskip('dolbat_interfaces.msg')
from dolbat_interfaces.msg import VideoFrame,LaneFrame,PathFrame
from dolbat_core.lane_detection import LaneFit,LaneResult
from dolbat_core.metric_planning import MetricBoundary,MetricPlan
from dolbat_core.week4_control import PurePursuitConfig,calculate_pure_pursuit
from dolbat_pipeline.codecs import encode_lanes,decode_lanes,encode_plan,decode_plan,encode_control,decode_control,points


def source():
    msg=VideoFrame();msg.frame_index=17;msg.source_fps=24.;msg.header.frame_id='camera';msg.header.stamp.sec=123
    return msg


def test_actual_lane_rows_and_partial_metadata_round_trip_exactly():
    pixels=np.array([[100,400],[101,401],[102,402]],float)
    fit=LaneFit(np.array([.01,.2,25.]),pixels,.7,.3,400.,402.,False)
    fit.support={'rows':3,'support_fraction':.75}
    lanes=LaneResult(fit,None,'INFERRED_RIGHT','only measured left',.8,np.zeros((8,8,3),np.uint8),np.zeros(8))
    lanes.partial_left=None;lanes.partial_right=fit
    lanes.partial_summary={'right':{'rows':3,'kind':'PARTIAL'}};lanes.context_rejections=['test']
    message=encode_lanes(lanes,source());decoded=decode_lanes(message)
    np.testing.assert_array_equal(decoded.left.pixels,pixels)
    np.testing.assert_array_equal(decoded.partial_right.coefficients,fit.coefficients)
    assert decoded.left.support==fit.support and decoded.partial_summary==lanes.partial_summary
    assert decoded.usable and decoded.context_rejections==['test']
    assert message.frame_index==17 and message.header.stamp.sec==123


def test_path_and_render_models_round_trip_without_refitting():
    x=np.linspace(20,200,10);path=np.column_stack((x,.001*x*x))
    rows=path+np.array([0.,45.])
    boundary=MetricBoundary(np.array([.001,0.,45.]),rows,rows,np.ones(10,bool),(20.,200.),0.,0.)
    plan=MetricPlan('JOINT_TWO_SIDES','actual rows',boundary,None,path,np.zeros_like(path),None,None,np.zeros(10),'joint')
    plan.full_model_left_cm=rows;plan.full_model_right_cm=path-[0.,45.]
    plan.partial_points_cm={'right':path[:3]};plan.joint_fit_accepted=True;plan.max_path_adjustment_cm=.4
    origin=LaneFrame();original=source();origin.header=original.header
    origin.frame_index=original.frame_index;origin.source_fps=original.source_fps;origin.status='DETECTED'
    message=encode_plan(plan,origin);decoded=decode_plan(message)
    np.testing.assert_array_equal(decoded.path_cm,path)
    np.testing.assert_array_equal(decoded.full_model_left_cm,rows)
    np.testing.assert_array_equal(decoded.full_model_right_cm,plan.full_model_right_cm)
    np.testing.assert_array_equal(decoded.partial_points_cm['right'],path[:3])
    assert decoded.left is not None and decoded.right is None
    assert decoded.joint_fit_accepted and decoded.max_path_adjustment_cm==.4
    assert message.perception_status=='DETECTED' and message.frame_index==17


@pytest.mark.parametrize('lateral',[0.,.3,-.3])
def test_control_round_trip_preserves_units_sign_and_intersection(lateral):
    path=np.array([[.5,lateral],[2.,lateral]])
    result=calculate_pure_pursuit(path,PurePursuitConfig())
    assert result.valid
    decoded=decode_control(encode_control(result,source()))
    assert decoded==result


def test_invalid_control_does_not_turn_ros_default_zeroes_into_a_valid_angle():
    result=calculate_pure_pursuit(np.array([[.1,0.],[.2,0.]]))
    assert not result.valid
    message=encode_control(result,source())
    assert not message.valid and message.segment_index==-1
    decoded=decode_control(message)
    assert decoded.steering_deg is None and decoded.target_m is None


def test_bad_coordinate_payload_is_rejected():
    with pytest.raises(ValueError,match='pairs'):points([1.,2.,3.])
    with pytest.raises(ValueError,match='nonfinite'):points([1.,np.nan])

