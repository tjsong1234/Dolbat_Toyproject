from pathlib import Path
import numpy as np
import pytest
import yaml
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.metric_planning import MetricPlan,MetricBoundary
from dolbat_core.stable_metric_planning import StableMetricLanePlanner,sampled_curvature

ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/'dolbat_core/resources/config/week3.yaml').read_text(encoding='utf-8'))['planning']
MAPPING=MetricMapping(.25,.5,319.5,639)


def planner(**overrides):
    return StableMetricLanePlanner(MAPPING,{**CONFIG,'path_smoothing_time_constant_s':.075,
        'max_path_adjustment_cm':4.,'path_blend_edge_cm':100.,**overrides},90,fps=24)


def path(offset=0,lo=20,hi=240,quadratic=0,status='OBSERVED_PAIR'):
    x=np.linspace(lo,hi,160);y=quadratic*x*x+offset;xy=np.column_stack((x,y))
    curvature=2*quadratic/(1+(2*quadratic*x)**2)**1.5
    def boundary(lateral_offset):
        points=xy+np.array([0.,lateral_offset])
        return MetricBoundary(np.array([quadratic,0.,offset+lateral_offset]),points,points,
            np.ones(len(points),bool),(lo,hi),0.,0.)
    left=boundary(45.) if status in ('OBSERVED_PAIR','ESTIMATED_FROM_LEFT') else None
    right=boundary(-45.) if status in ('OBSERVED_PAIR','ESTIMATED_FROM_RIGHT') else None
    return MetricPlan(status,'test',left,right,xy,MAPPING.cm_to_pixels(xy),
        np.array([quadratic,0.,offset]) if status=='OBSERVED_PAIR' else None,None,curvature,'test')


def test_missing_current_path_clears_history_instead_of_replaying_it():
    p=planner();p.stabilize(path(0))
    blank=MetricPlan('NO_OBSERVATION','missing',None,None,np.empty((0,2)),np.empty((0,2)),None,None,np.array([]),'none')
    lost=p.stabilize(blank)
    assert not lost.usable and len(lost.raw_path_cm)==0 and p.previous_path_cm is None
    assert p.previous_raw_range_cm is None and p.previous_status is None
    recovered=p.stabilize(path(25))
    assert not recovered.stabilization_applied
    np.testing.assert_array_equal(recovered.path_cm,recovered.raw_path_cm)


def test_correction_is_capped_and_raw_current_provenance_is_preserved():
    p=planner();p.stabilize(path(0));raw=path(20,status='ESTIMATED_FROM_RIGHT');r=p.stabilize(raw)
    assert r.status==raw.status and r.left is raw.left and r.right is raw.right
    assert r.stabilization_applied and r.max_path_adjustment_cm==pytest.approx(4)
    np.testing.assert_array_equal(r.raw_path_cm,raw.path_cm)
    np.testing.assert_array_equal(r.path_cm[:,0],raw.path_cm[:,0])
    assert np.max(np.linalg.norm(r.path_cm-r.raw_path_cm,axis=1))<=4+1e-10
    assert np.all((r.path_cm[:,1]>0)&(r.path_cm[:,1]<20))


def test_real_turn_is_followed_with_bounded_lag_instead_of_frozen_history():
    p=planner();positions=[]
    for offset in range(0,61,5):
        r=p.stabilize(path(offset));positions.append(np.median(r.path_cm[:,1]))
        assert np.max(abs(r.path_cm[:,1]-offset))<=4+1e-9
    assert np.all(np.diff(positions)>0) and positions[-1]>=56


def test_new_ranges_stay_current_and_do_not_extrapolate_old_path():
    p=planner(path_pair_range_reset_cm=0);p.stabilize(path(0,lo=80,hi=180))
    raw=path(10,lo=20,hi=240);r=p.stabilize(raw)
    inside=(raw.path_cm[:,0]>=80)&(raw.path_cm[:,0]<=180)
    np.testing.assert_array_equal(r.path_cm[~inside],raw.path_cm[~inside])
    assert r.path_cm[0,0]==20 and r.path_cm[-1,0]==240
    assert r.shared_range_cm==(80,180)
    assert np.all(np.isfinite(r.curvature_per_cm))
    assert r.center_coefficients is None and r.raw_center_coefficients is not None


def test_disjoint_current_range_resets_to_current_geometry():
    p=planner(path_pair_range_reset_cm=0);p.stabilize(path(0,lo=20,hi=90));raw=path(30,lo=150,hi=240)
    r=p.stabilize(raw)
    np.testing.assert_array_equal(r.path_cm,raw.path_cm)
    assert not r.stabilization_applied and r.stabilization_reset_reason=='no_shared_current_range'


def test_direct_observed_side_flip_does_not_blend_different_estimates():
    p=planner();p.stabilize(path(0,status='ESTIMATED_FROM_LEFT'))
    raw=path(8,status='ESTIMATED_FROM_RIGHT');r=p.stabilize(raw)
    np.testing.assert_array_equal(r.path_cm,raw.path_cm)
    assert not r.stabilization_applied and r.stabilization_reset_reason=='observed_side_changed'


def test_paired_support_range_change_prefers_new_raw_geometry_and_updates_history():
    p=planner(path_pair_range_reset_cm=20)
    p.stabilize(path(0));assert p.stabilize(path(20)).stabilization_applied
    raw=path(24,lo=30,hi=280);r=p.stabilize(raw)
    np.testing.assert_array_equal(r.path_cm,raw.path_cm)
    assert not r.stabilization_applied
    assert r.stabilization_reset_reason=='paired_observation_range_changed'
    assert p.previous_raw_range_cm==(30,280)
    assert p.stabilize(path(25,lo=30,hi=280)).stabilization_applied


def test_range_change_in_single_boundary_estimate_keeps_bounded_shared_filter():
    p=planner(path_pair_range_reset_cm=20)
    p.stabilize(path(0,status='ESTIMATED_FROM_RIGHT'))
    raw=path(10,hi=280,status='ESTIMATED_FROM_RIGHT');r=p.stabilize(raw)
    assert r.stabilization_applied and r.status==raw.status
    assert r.stabilization_reset_reason!='paired_observation_range_changed'
    new=r.path_cm[:,0]>240
    np.testing.assert_array_equal(r.path_cm[new],raw.path_cm[new])


def test_stabilized_curvature_uses_final_cm_path_and_retains_turn_sign():
    p=planner();p.stabilize(path(0,quadratic=.002))
    r=p.stabilize(path(8,quadratic=.002))
    x,y=r.path_cm.T;dy=np.gradient(y,x,edge_order=2);ddy=np.gradient(dy,x,edge_order=2)
    np.testing.assert_allclose(r.curvature_per_cm,ddy/(1+dy*dy)**1.5,atol=1e-12)
    assert np.all(r.curvature_per_cm>0)
    np.testing.assert_allclose(r.path_uv,MAPPING.cm_to_pixels(r.path_cm),atol=1e-12)
    np.testing.assert_allclose(sampled_curvature(path(quadratic=-.002).path_cm),
        -sampled_curvature(path(quadratic=.002).path_cm),atol=1e-10)


def test_invalid_stabilization_settings_are_rejected():
    with pytest.raises(ValueError):planner(max_path_adjustment_cm=-1)
    with pytest.raises(ValueError):StableMetricLanePlanner(MAPPING,CONFIG,90,fps=0)
