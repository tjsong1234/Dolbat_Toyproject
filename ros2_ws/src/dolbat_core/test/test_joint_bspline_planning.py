from pathlib import Path
import numpy as np
import yaml
from dolbat_core.joint_bspline_planning import JointBSplineLanePlanner
from test_yellow_guided_planning import MAPPING,observed,lanes
import json
import pytest
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.lane_detection import LaneFit,LaneResult

CFG=yaml.safe_load((Path(__file__).parents[1]/'dolbat_core/resources/config/week3_joint.yaml').read_text(encoding='utf-8'))['planning']


def test_joint_model_uses_actual_left_beyond_short_right_support():
    result=JointBSplineLanePlanner(MAPPING,CFG,90).plan(lanes(observed(lambda x:45+0*x,20,280),observed(lambda x:-45+0*x,20,130)))
    assert result.joint_fit_accepted and result.path_cm[-1,0]>270
    assert result.path_cm[-1,0]>result.right.interval_cm[1]+100
    np.testing.assert_allclose(result.path_cm[:,1],0,atol=.02)
    np.testing.assert_allclose((result.full_model_left_cm+result.full_model_right_cm)/2,result.raw_path_cm,atol=1e-10)
    assert result.full_model_left_inferred_mask.all() and result.full_model_right_inferred_mask.all()


def test_stationary_road_switching_observed_side_does_not_move_joint_center():
    planner=JointBSplineLanePlanner(MAPPING,CFG,90)
    a=planner.plan(lanes(None,observed(lambda x:-45+0*x)))
    b=planner.plan(lanes(observed(lambda x:45+0*x),None))
    assert a.joint_fit_accepted and b.joint_fit_accepted
    np.testing.assert_allclose(a.path_cm,b.path_cm,atol=.02)
    assert b.stabilization_context['reference_side']=='none'
    assert b.stabilization_reset_reason!='observed_side_changed'


def test_current_right_partial_is_separate_weak_evidence_not_full_detection():
    observation=lanes(observed(lambda x:45+0*x),None)
    observation.partial_right=observed(lambda x:-45+0*x,80,130)
    points=observation.partial_right.pixels.copy()
    result=JointBSplineLanePlanner(MAPPING,CFG,90).plan(observation)
    assert result.joint_fit_accepted and result.right is None
    assert result.status=='JOINT_PARTIAL_SUPPORT'
    assert any(e['side']=='right' and e['partial'] for e in result.joint_evidence)
    np.testing.assert_array_equal(points,observation.partial_right.pixels)


def test_no_current_rows_do_not_replay_a_joint_spline():
    planner=JointBSplineLanePlanner(MAPPING,CFG,90)
    assert planner.plan(lanes(None,observed(lambda x:-45+0*x))).usable
    lost=planner.plan(lanes(None,None))
    assert not lost.usable and not lost.joint_fit_accepted
    assert lost.status.startswith('FALLBACK_')
    assert planner.previous_path_cm is None


def test_outlier_partial_is_removed_from_joint_initialisation_and_input():
    observation=lanes(None,observed(lambda x:-45+0*x))
    observation.partial_left=observed(lambda x:190+0*x,100,130)
    result=JointBSplineLanePlanner(MAPPING,CFG,90).plan(observation)
    assert result.joint_fit_accepted
    assert result.joint_rejected_partials
    assert not result.partial_used
    assert not any(e['partial'] for e in result.joint_evidence)
    np.testing.assert_allclose(result.raw_path_cm[:,1],0,atol=.02)


def test_rejected_joint_fit_does_not_reuse_old_fallback_temporal_state():
    planner=JointBSplineLanePlanner(MAPPING,CFG,90)
    planner.fallback.previous_path_cm=np.array([[20.,999.],[260.,999.]])
    planner.fallback.previous_context={'reference_side':'right','both_observed':False,'observation_range_cm':(20,260)}
    result=planner.plan(lanes(observed(lambda x:190+0*x),observed(lambda x:-45+0*x)))
    assert result.usable and not result.joint_fit_accepted and result.status.startswith('FALLBACK_')
    assert result.max_path_adjustment_cm==0
    np.testing.assert_allclose(result.path_cm[:,1],0,atol=.02)


VIDEO_ROWS=json.loads((Path(__file__).parent/'fixtures/week3_bend_observations.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case',VIDEO_ROWS['cases'],ids=lambda c:str(c['frame']))
def test_saved_actual_video_rows_joint_fit_remains_smooth_and_current(case):
    mapping=MetricMapping(**VIDEO_ROWS['mapping'])
    def observation(side):
        px=mapping.cm_to_pixels(np.asarray(case['observed_row_centers_cm'][side]))
        px[:,1]=np.rint(px[:,1])
        return LaneFit(np.polyfit(px[:,1],px[:,0],2),px,0,float(np.ptp(px[:,1])/639),float(px[:,1].min()),float(px[:,1].max()))
    result=LaneResult(observation('left'),observation('right'),'DETECTED','saved actual row-centre summaries',1,
        np.zeros((640,800,3),np.uint8),np.zeros(800))
    plan=JointBSplineLanePlanner(mapping,CFG,90).plan(result)
    assert plan.joint_fit_accepted and plan.usable
    assert plan.joint_diagnostics['generated_observation_rows']==0
    # Scene regression bound, not a vehicle's physical curvature constraint.
    assert np.max(abs(plan.raw_curvature_per_cm))<.01
    assert plan.boundary_pair_midpoint_error_cm<1e-10
