from pathlib import Path
import numpy as np
import pytest
import yaml
from dolbat_core.metric_mapping import MetricMapping,physical_arc_length,solve_forward_scale
from dolbat_core.metric_planning import MetricBoundary,MetricLanePlanner,fit_metric_boundary,normal_offset
from dolbat_core.lane_detection import LaneFit,LaneResult

ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/'dolbat_core/resources/config/week3.yaml').read_text(encoding='utf-8'))['planning']


def mapping():return MetricMapping(.25,.5,319.5,639)


def lane(u,inferred=False):
    vv=np.repeat(np.arange(80,601),9);uu=u+np.tile(np.arange(-4,5),521)
    px=np.column_stack((uu,vv)) if not inferred else np.empty((0,2))
    return LaneFit(np.array([0.,0.,u]),px,1,.8,80,600,inferred)


def result(left,right,status):
    return LaneResult(left,right,status,'test',1,np.zeros((640,640,3),np.uint8),np.zeros(640))


def test_cm_mapping_round_trip_and_axis_signs():
    m=mapping();uv=np.array([[0,0],[319.5,639],[639,400],[210.3,135.7]])
    np.testing.assert_allclose(m.cm_to_pixels(m.pixels_to_cm(uv)),uv,atol=1e-10)
    assert m.pixels_to_cm(uv)[0,0]>0 and m.pixels_to_cm(uv)[0,1]>0
    np.testing.assert_allclose(m.pixels_to_cm(uv)[1],[0,0])


def test_mapping_rejects_invalid_scale():
    with pytest.raises(ValueError):MetricMapping(0,.5,319.5,639)
    with pytest.raises(ValueError):MetricMapping(.2,float('nan'),319.5,639)


def test_forward_scale_uses_anisotropic_arc_length():
    co=np.array([.0004,.1,100]);length=physical_arc_length(co,100,240,.25,.5)
    assert solve_forward_scale(co,100,240,.25,length)==pytest.approx(.5,abs=1e-10)
    assert solve_forward_scale(np.array([0.,0.,100]),100,200,.25,50)==pytest.approx(.5)


def test_metric_fit_is_from_observed_points_and_unit_consistent():
    fit=fit_metric_boundary(lane(139.5),mapping(),CONFIG)
    assert fit is not None
    np.testing.assert_allclose(fit.y(np.array([20,100,200])),45,atol=1e-8)
    assert fit.rmse_cm<1e-8
    assert fit_metric_boundary(lane(499.5,True),mapping(),CONFIG) is None


def test_two_observed_functions_produce_center_without_extrapolation():
    p=MetricLanePlanner(mapping(),CONFIG,90)
    r=p.plan(result(lane(139.5),lane(499.5),'DETECTED'))
    assert r.status=='OBSERVED_PAIR' and r.method=='midpoint_quadratic'
    np.testing.assert_allclose(r.path_cm[:,1],0,atol=1e-8)
    assert r.path_cm[0,0]>=r.left.interval_cm[0] and r.path_cm[-1,0]<=r.right.interval_cm[1]


def test_single_observed_function_does_not_invent_scatter_points():
    p=MetricLanePlanner(mapping(),CONFIG,90)
    r=p.plan(result(lane(139.5),lane(499.5,True),'INFERRED_RIGHT'))
    assert r.status=='ESTIMATED_FROM_LEFT' and r.right is None and r.center_coefficients is None
    np.testing.assert_allclose(r.path_cm[:,1],0,atol=1e-8)
    assert r.inferred_boundary_cm is not None


def test_normal_offset_distance_and_tangent_are_consistent_on_curve():
    b=MetricBoundary(np.array([.002,.1,40]),np.empty((0,2)),np.empty((0,2)),np.array([],bool),(10,180),0,0)
    x=np.linspace(10,180,60);d=-45.;curve=np.column_stack((x,b.y(x)));offset=normal_offset(b,x,d)
    np.testing.assert_allclose(np.linalg.norm(offset-curve,axis=1),abs(d),atol=1e-10)
    tangent=np.column_stack((np.ones(len(x)),b.slope(x)))
    np.testing.assert_allclose(np.sum((offset-curve)*tangent,axis=1),0,atol=1e-10)


def test_empty_observation_does_not_reuse_a_previous_path():
    p=MetricLanePlanner(mapping(),CONFIG,90)
    assert p.plan(result(lane(139.5),lane(499.5),'DETECTED')).usable
    r=p.plan(result(None,None,'FAILED'))
    assert not r.usable and r.status=='NO_OBSERVATION'


def test_quadratic_coefficients_after_anisotropic_unit_conversion():
    m=mapping();v=np.repeat(np.arange(100,550),5)
    a,b,c=.0004,-.08,135.
    u=a*v*v+b*v+c+np.tile(np.arange(-2,3),450)
    observed=LaneFit(np.array([a,b,c]),np.column_stack((u,v)),1,.7,100,549)
    boundary=fit_metric_boundary(observed,m,CONFIG)
    expected=np.array([-m.lateral_cm_per_px*a/m.forward_cm_per_px**2,
        m.lateral_cm_per_px*(2*a*m.origin_v_px+b)/m.forward_cm_per_px,
        m.lateral_cm_per_px*(m.origin_u_px-a*m.origin_v_px**2-b*m.origin_v_px-c)])
    np.testing.assert_allclose(boundary.coefficients,expected,atol=1e-8)


def test_folded_or_backward_normal_offset_is_not_reported_as_valid_path():
    m=mapping();vv=np.repeat(np.arange(579,640),3);xx=m.forward_cm_per_px*(m.origin_v_px-vv)
    yy=-45+.05*xx**2;uu=m.origin_u_px-yy/m.lateral_cm_per_px+np.tile([-1,0,1],61)
    observed=LaneFit(np.zeros(3),np.column_stack((uu,vv)),1,.2,579,639)
    r=MetricLanePlanner(m,CONFIG,90).plan(result(lane(100,True),observed,'INFERRED_LEFT'))
    assert not r.usable and r.status=='INVALID_OFFSET'
