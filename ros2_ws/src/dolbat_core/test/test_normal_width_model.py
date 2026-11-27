import numpy as np
from test_yellow_guided_planning import observed,lanes,planner,MAPPING,CONFIG
from dolbat_core.stable_metric_planning import sampled_curvature


def test_short_observed_support_has_no_local_transition_s_bend():
    p=planner().plan(lanes(observed(lambda x:51+0*x,100,140),observed(lambda x:-45+0*x)))
    assert p.usable and p.width_evidence_kind=='full'
    assert 90<p.width_model_cm<96
    assert max(abs(sampled_curvature(p.full_model_left_cm)))<1e-10
    assert max(abs(p.raw_curvature_per_cm))<1e-10
    assert p.full_model_left_inferred_mask.all()


def test_curve_constant_normal_distance_and_analytic_midpoint_curvature():
    from dolbat_core.yellow_guided_planning import YellowGuidedLanePlanner
    p=YellowGuidedLanePlanner(MAPPING,{**CONFIG,'blue_width_allow_affine':False},90).plan(lanes(observed(lambda x:45+.001*x*x,100,170),observed(lambda x:-45+.001*x*x)))
    t=p.full_model_parameter_cm;k=p.right.curvature(t)
    np.testing.assert_allclose(np.linalg.norm(p.full_model_left_cm-p.full_model_right_cm,axis=1),p.width_model_cm,atol=1e-10)
    expected=k/(1-.5*p.width_model_cm*k)
    # Central numerical derivatives approximate the analytic normal offset.
    np.testing.assert_allclose(p.raw_curvature_per_cm[2:-2],expected[2:-2],rtol=2e-4,atol=1e-6)
    assert max(abs(p.raw_curvature_per_cm))<.004


def test_partial_real_rows_have_bounded_influence_without_detection_promotion():
    r=lanes(None,observed(lambda x:-45+0*x));r.partial_left=observed(lambda x:35+0*x,100,130)
    saved=r.partial_left.pixels.copy();p=planner().plan(r)
    assert p.partial_used and p.width_evidence_kind=='partial'
    assert abs(p.width_model_cm-90)<=CONFIG['blue_partial_max_adjustment_cm']
    assert p.left is None and r.status=='INFERRED_LEFT'
    np.testing.assert_array_equal(saved,r.partial_left.pixels)
    assert p.full_model_left_inferred_mask.all()


def test_implausible_partial_width_or_direction_cannot_pull_model():
    for fn in [lambda x:170+0*x,lambda x:45+1.2*(x-100)]:
        r=lanes(None,observed(lambda x:-45+0*x));r.partial_left=observed(fn,100,130)
        p=planner().plan(r)
        assert not p.partial_used and p.width_model_cm==90


def test_global_fit_uses_actual_row_centers_not_extrapolated_observation():
    p=planner().plan(lanes(observed(lambda x:49+0*x,100,130),observed(lambda x:-45+0*x)))
    assert p.width_support_span_cm<=30.00001
    assert p.width_accepted_rows==61
    assert p.path_cm[0,0]<100 and p.path_cm[-1,0]>130
