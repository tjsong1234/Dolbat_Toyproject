from pathlib import Path
import numpy as np
import yaml
from dolbat_core.lane_detection import LaneFit,LaneResult
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.yellow_guided_planning import YellowGuidedLanePlanner

ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/'dolbat_core/resources/config/week3_final.yaml').read_text(encoding='utf-8'))['planning']
MAPPING=MetricMapping(.25,.5,399.5,639)


def observed(y_fn,lo=20,hi=260):
    v=np.arange(round(639-hi/.5),round(639-lo/.5)+1)
    x=(639-v)*.5;u=399.5-y_fn(x)/.25
    vv=np.repeat(v,5);uu=np.repeat(u,5)+np.tile(np.arange(-2,3),len(v))
    pixels=np.column_stack((uu,vv));co=np.polyfit(v,u,2)
    return LaneFit(co,pixels,0,(v.max()-v.min())/639,float(v.min()),float(v.max()))


def lanes(left,right):
    status='DETECTED' if left is not None and right is not None else ('INFERRED_LEFT' if right is not None else 'INFERRED_RIGHT' if left is not None else 'FAILED')
    return LaneResult(left,right,status,'test',1,np.zeros((640,800,3),np.uint8),np.zeros(800))


def planner():return YellowGuidedLanePlanner(MAPPING,CONFIG,90,fps=24)


def test_short_left_support_does_not_shorten_blue_or_center_to_left_interval():
    right=observed(lambda x:np.full_like(x,-45.));left=observed(lambda x:np.full_like(x,45.),90,150)
    p=planner().plan(lanes(left,right))
    assert p.usable and p.boundary_pair_available
    np.testing.assert_allclose(p.full_model_parameter_cm[[0,-1]],[20,260],atol=1e-8)
    np.testing.assert_allclose(p.full_model_left_cm[:,0],p.full_model_right_cm[:,0],atol=1e-8)
    assert p.path_cm[0,0]<90 and p.path_cm[-1,0]>150 and np.ptp(p.path_cm[:,0])>230
    outside=(p.full_model_parameter_cm<90)|(p.full_model_parameter_cm>150)
    assert p.full_model_left_inferred_mask[outside].all()
    np.testing.assert_allclose(p.full_model_left_cm[outside,1],45,atol=1e-8)
    np.testing.assert_allclose((p.boundary_pair_left_cm+p.boundary_pair_right_cm)/2,p.raw_path_cm,atol=1e-10)


def test_blue_continues_without_any_current_left_pixels_and_follows_new_yellow_slope():
    p=planner();a=p.plan(lanes(None,observed(lambda x:-45+0*x)))
    b=p.plan(lanes(None,observed(lambda x:-45+.25*x)))
    assert b.left is None and b.boundary_pair_left_estimated and b.full_model_left_inferred_mask.all()
    assert len(b.full_model_left_cm)==CONFIG['samples']
    assert np.median(np.diff(a.full_model_left_cm[:,1]))<1e-8
    assert np.median(np.diff(b.full_model_left_cm[:,1]))>.2
    np.testing.assert_allclose(np.linalg.norm(b.full_model_left_cm-b.full_model_right_cm,axis=1),90,atol=1e-8)


def test_missing_left_interval_changes_do_not_change_yellow_guided_coverage():
    p=planner();right=observed(lambda x:-45+.001*x*x)
    a=p.plan(lanes(observed(lambda x:45+.001*x*x,70,150),right))
    b=p.plan(lanes(None,right))
    c=p.plan(lanes(observed(lambda x:45+.001*x*x,110,190),right))
    assert all(plan.usable for plan in [a,b,c])
    np.testing.assert_array_equal(a.full_model_parameter_cm,b.full_model_parameter_cm)
    np.testing.assert_array_equal(b.full_model_parameter_cm,c.full_model_parameter_cm)
    # Fitting normal width can change X as well as Y on a sloping reference.
    # Coverage is defined by the unchanged CURRENT yellow parameter interval.
    assert all(np.all(np.diff(q.full_model_left_cm[:,0])>0) for q in (a,b,c))


def test_current_left_data_are_not_modified_or_invented_when_blue_is_completed():
    right=observed(lambda x:-45+0*x);left=observed(lambda x:49+0*x,70,190)
    points=left.pixels.copy();p=planner().plan(lanes(left,right))
    np.testing.assert_array_equal(left.pixels,points)
    assert p.left is not None and len(p.left.points_cm)==len(points)
    inside=(p.full_model_parameter_cm>100)&(p.full_model_parameter_cm<160)
    assert 48<np.mean(p.full_model_left_cm[inside,1])<49
    outside=(p.full_model_parameter_cm<70)|(p.full_model_parameter_cm>190)
    np.testing.assert_allclose(p.full_model_left_cm[outside,1],p.full_model_left_cm[inside,1].mean(),atol=1e-8)
    assert p.full_model_left_inferred_mask.all()


def test_implausible_left_object_cannot_pull_yellow_guided_blue_model():
    right=observed(lambda x:-45+0*x);left=observed(lambda x:190+0*x,70,190)
    p=planner().plan(lanes(left,right))
    assert p.usable and p.full_model_left_inferred_mask.all()
    np.testing.assert_allclose(p.full_model_left_cm[:,1],45,atol=1e-8)


def test_no_current_observations_clears_path_and_completed_blue_history():
    p=planner();assert p.plan(lanes(None,observed(lambda x:-45+0*x))).usable
    lost=p.plan(lanes(None,None))
    assert not lost.usable and not lost.boundary_pair_available
    assert len(lost.full_model_left_cm)==0 and p.previous_path_cm is None


def test_left_only_current_observation_uses_mirrored_fallback_without_yellow_claim():
    p=planner().plan(lanes(observed(lambda x:45+0*x),None))
    assert p.usable and not p.yellow_guided and p.full_model_reference_side=='left'
    assert p.boundary_pair_right_estimated and p.right is None


def test_complete_blue_model_remains_available_even_when_forward_center_is_clipped():
    p=planner().plan(lanes(None,observed(lambda x:-45+2*x,1,100)))
    assert p.usable and p.status.endswith('_TRUNCATED')
    assert len(p.full_model_left_cm)==CONFIG['samples']
    assert len(p.boundary_pair_left_cm)<len(p.full_model_left_cm)
    np.testing.assert_allclose(p.full_model_parameter_cm[[0,-1]],[1,100],atol=1e-8)
    assert p.path_cm[0,0]>=0


def test_singular_full_normal_offset_is_rejected_instead_of_replaying_old_blue():
    p=planner();assert p.plan(lanes(None,observed(lambda x:-45+0*x))).usable
    bad=p.plan(lanes(None,observed(lambda x:-45+.05*x*x,0,60)))
    assert not bad.usable and bad.status=='INVALID_BLUE_OFFSET'
    assert p.previous_path_cm is None and len(bad.full_model_left_cm)==0
