from pathlib import Path
import numpy as np
import yaml
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.metric_planning import MetricBoundary,MetricPlan,normal_offset
from dolbat_core.boundary_pair_planning import attach_boundary_pair

ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/'dolbat_core/resources/config/week3.yaml').read_text(encoding='utf-8'))['planning']
MAPPING=MetricMapping(.25,.5,399.5,639)


def boundary(co,lo=10.,hi=180.):
    return MetricBoundary(np.asarray(co,dtype=float),np.array([[10.,20.],[20.,21.]]),
        np.array([[10.,20.],[20.,21.]]),np.array([True,True]),(lo,hi),0.,0.)


def one_side(side='right',co=(.001,.1,-45.),truncated=False):
    b=boundary(co);x=np.linspace(10,180,160);d=45 if side=='right' else -45
    raw=normal_offset(b,x,d);partner=normal_offset(b,x,2*d)
    if truncated:raw,partner=raw[25:120],partner[25:120]
    p=MetricPlan('ESTIMATED_FROM_'+side.upper(),'',b if side=='left' else None,b if side=='right' else None,
        raw.copy(),MAPPING.cm_to_pixels(raw),None,partner,np.zeros(len(raw)),'normal_offset_parametric')
    p.raw_path_cm=raw.copy()
    return p


def test_estimated_left_and_observed_right_have_exact_vector_midpoint_on_curve():
    p=one_side();actual=p.right;points=actual.points_cm.copy()
    attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert p.boundary_pair_available and p.boundary_pair_left_estimated
    assert not p.boundary_pair_right_estimated and p.left is None and p.right is actual
    np.testing.assert_array_equal(actual.points_cm,points)
    np.testing.assert_allclose((p.boundary_pair_left_cm+p.boundary_pair_right_cm)/2,p.raw_path_cm,atol=1e-12)
    np.testing.assert_allclose(p.boundary_pair_right_cm[:,1],actual.y(p.boundary_pair_right_cm[:,0]),atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(p.boundary_pair_left_cm-p.boundary_pair_right_cm,axis=1),90,atol=1e-12)
    np.testing.assert_allclose(p.boundary_pair_left_uv,MAPPING.cm_to_pixels(p.boundary_pair_left_cm))


def test_inferred_right_pair_uses_opposite_normal_and_preserves_observation():
    p=one_side('left',(.001,.1,45.));attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert p.boundary_pair_available and p.boundary_pair_right_estimated
    assert p.boundary_pair_provenance=='observed_left_plus_estimated_right'
    np.testing.assert_allclose((p.boundary_pair_left_cm+p.boundary_pair_right_cm)/2,p.raw_path_cm,atol=1e-12)


def test_truncated_normal_parameter_range_is_not_replaced_by_full_interval():
    p=one_side(truncated=True);attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert p.boundary_pair_available and len(p.boundary_pair_left_cm)==95
    assert p.boundary_pair_right_cm[0,0]>p.right.interval_cm[0]
    assert p.boundary_pair_right_cm[-1,0]<p.right.interval_cm[1]
    np.testing.assert_allclose(p.boundary_pair_midpoint_cm,p.raw_path_cm,atol=1e-12)


def test_two_observed_functions_use_the_current_shared_forward_range():
    left=boundary((.001,.1,45),20,160);right=boundary((.001,.1,-45),10,180)
    x=np.linspace(20,160,100);raw=np.column_stack((x,(left.y(x)+right.y(x))/2))
    p=MetricPlan('OBSERVED_PAIR','',left,right,raw,MAPPING.cm_to_pixels(raw),(left.coefficients+right.coefficients)/2,None,np.zeros(100),'midpoint_quadratic')
    attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert p.boundary_pair_available and not p.boundary_pair_left_estimated and not p.boundary_pair_right_estimated
    assert p.boundary_pair_provenance=='two_observed_functions'
    np.testing.assert_allclose(p.boundary_pair_midpoint_cm,raw)


def test_temporal_path_is_not_misrepresented_as_the_exact_raw_midpoint():
    p=one_side();p.path_cm[:,1]+=.8;final=p.path_cm.copy()
    attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert p.boundary_pair_available
    np.testing.assert_array_equal(p.path_cm,final)
    assert abs(p.boundary_pair_display_path_offset_cm-.8)<1e-10
    assert p.boundary_pair_midpoint_error_cm<1e-12


def test_no_observation_and_missing_partner_never_create_boundary_pair():
    p=one_side();p.inferred_boundary_cm=None;attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert not p.boundary_pair_available and len(p.boundary_pair_left_cm)==0
    empty=np.empty((0,2));p=MetricPlan('NO_OBSERVATION','',None,None,empty,empty,None,None,np.array([]),'none')
    attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert not p.boundary_pair_available and len(p.boundary_pair_midpoint_cm)==0


def test_corrupted_or_singular_partner_is_not_displayed_as_valid_boundary():
    p=one_side();p.inferred_boundary_cm[:,1]+=.5;attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert not p.boundary_pair_available
    p=one_side(co=(.007,0,-45));attach_boundary_pair(p,MAPPING,CONFIG,90)
    assert not p.boundary_pair_available and 'folds' in p.boundary_pair_reason
