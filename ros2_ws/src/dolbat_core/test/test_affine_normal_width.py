"""A low-degree width model must earn its extra degree of freedom."""
import numpy as np
import pytest
from dolbat_core.metric_planning import MetricBoundary
from dolbat_core.normal_width_model import estimate_normal_width,NormalWidthEstimate

CONFIG={'offset_singularity_margin':.05,'blue_width_allow_affine':True,
    'blue_width_prior_support_cm':30.,'blue_width_affine_prior_support_cm':30.,
    'blue_width_max_adjustment_cm':12.,'blue_width_affine_max_gradient':.10}


def boundary(co,lo=0.,hi=240.,samples=241):
    x=np.linspace(lo,hi,samples);co=np.asarray(co,dtype=float)
    xy=np.column_stack((x,np.polyval(co,x)))
    return MetricBoundary(co,xy,xy,np.ones(len(x),bool),(lo,hi),0.,0.)


def estimate(observation,reference=None,partial=False,**overrides):
    return estimate_normal_width(reference or boundary([0.,0.,-45.]),observation,90.,
        {**CONFIG,**overrides},partial=partial)


def test_broad_linear_distance_support_can_select_affine_and_reduce_actual_residual():
    fit=estimate(boundary([0.,.06,37.8]))
    assert fit.model=='affine' and fit.kind=='full'
    assert 0.<fit.slope_cm_per_cm<.06
    assert fit.reference_midpoint_cm==120. and fit.reference_scale_cm==120.
    np.testing.assert_allclose(fit.evaluate([0.,240.]),
        fit.width_cm+fit.slope_cm_per_cm*np.array([-120.,120.]),atol=1e-12)
    assert fit.model_rmse_cm<=.8*fit.constant_rmse_cm
    assert fit.constant_rmse_cm-fit.model_rmse_cm>=.3


def test_affine_is_disabled_by_default_and_partial_never_uses_it():
    observation=boundary([0.,.06,37.8])
    default=estimate(observation,blue_width_allow_affine=False)
    partial=estimate(observation,partial=True,blue_partial_max_scatter_cm=10.)
    assert default.model==partial.model=='constant'
    assert default.slope_cm_per_cm==partial.slope_cm_per_cm==0.
    assert abs(partial.width_cm-90.)<=1.5


@pytest.mark.parametrize('lo,hi',[(100.,160.),(120.,200.),(0.,70.)])
def test_short_or_narrow_fraction_support_keeps_constant(lo,hi):
    fit=estimate(boundary([0.,.06,37.8],lo,hi),reference=boundary([0.,0.,-45.],0.,400.))
    assert fit.model=='constant' and fit.slope_cm_per_cm==0.


def test_parallel_rows_do_not_select_a_spurious_width_gradient():
    fit=estimate(boundary([0.,0.,49.]))
    assert fit.model=='constant' and fit.slope_cm_per_cm==0.
    np.testing.assert_allclose(fit.evaluate([0.,120.,240.]),fit.width_cm)


def test_global_endpoint_and_gradient_limits_scale_coefficients_without_point_clipping():
    fit=estimate(boundary([0.,.10,40.]),
        blue_width_max_adjustment_cm=3.,blue_width_affine_max_gradient=.02,
        blue_width_affine_prior_support_cm=.01,
        blue_width_affine_min_relative_improvement=0.,
        blue_width_affine_min_absolute_improvement_cm=0.)
    # The proposal may be rejected if the globally limited affine model no
    # longer improves the constant model. Either result must obey both bounds.
    endpoints=fit.evaluate([0.,240.])
    assert np.max(abs(endpoints-90.))<=3.+1e-10
    assert abs(fit.slope_cm_per_cm)<=.02+1e-12
    grid=np.linspace(0.,240.,100)
    np.testing.assert_allclose(np.diff(fit.evaluate(grid)),
        fit.slope_cm_per_cm*np.diff(grid),atol=1e-12)


def test_affine_normal_distance_fit_has_cm_translation_invariance():
    first=estimate(boundary([0.,.06,37.8]))
    shift=1000.
    second=estimate(boundary([0.,.06,37.8-.06*shift],shift,shift+240.),
        reference=boundary([0.,0.,-45.],shift,shift+240.))
    assert first.model==second.model=='affine'
    assert second.width_cm==pytest.approx(first.width_cm,abs=1e-10)
    assert second.slope_cm_per_cm==pytest.approx(first.slope_cm_per_cm,abs=1e-12)
    np.testing.assert_allclose(second.evaluate(np.array([0.,100.,240.])+shift),
        first.evaluate([0.,100.,240.]),atol=1e-10)


def test_global_regularization_prior_cannot_be_overridden_by_duplicate_row_count():
    coarse=estimate(boundary([0.,.06,37.8],samples=121))
    dense=estimate(boundary([0.,.06,37.8],samples=481))
    assert coarse.model==dense.model=='affine'
    assert abs(coarse.slope_cm_per_cm-dense.slope_cm_per_cm)<.001
    assert abs(coarse.width_cm-dense.width_cm)<.01


def test_empty_evidence_evaluates_nominal_over_reference_without_fake_rows():
    fit=estimate(None,reference=boundary([0.,0.,-45.],20.,280.))
    assert fit.kind=='none' and fit.model=='constant' and fit.accepted_rows==0
    assert fit.reference_midpoint_cm==150.
    np.testing.assert_array_equal(fit.evaluate([20.,150.,280.]),[90.,90.,90.])


def test_default_metadata_remains_compatible_with_constant_estimate_callers():
    fit=NormalWidthEstimate(92.,'full','old caller',np.array([]),np.array([]),np.array([]))
    assert fit.model=='constant' and fit.slope_cm_per_cm==0.
    np.testing.assert_array_equal(fit.evaluate([0.,100.]),[92.,92.])
