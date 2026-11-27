"""Geometric invariants that must survive reuse of computed spline bases."""
import numpy as np
from dolbat_core.bspline_basis import clamped_knots
from dolbat_core.joint_lane_model import JointLaneModel


def model(knots=None,coefficients=None):
    return JointLaneModel(np.array([3.,5.,12.,20.,17.,10.,6.]) if coefficients is None else coefficients,
                          clamped_knots((0.,100.),7) if knots is None else knots,
                          (0.,100.),90.,0.)


def test_coefficient_translation_after_basis_reuse_translates_the_curve():
    m=model();t=np.linspace(0,100,47);before=m.center(t)
    m.coefficients+=7.5
    after=m.center(t)
    np.testing.assert_allclose(after[:,0],before[:,0],atol=0)
    np.testing.assert_allclose(after[:,1]-before[:,1],7.5,atol=1e-12)


def test_in_place_knot_and_parameter_edits_match_a_fresh_model():
    m=model();t=np.linspace(0,100,47)
    m.center(t);m.curvature(t)
    m.knots[4]=m.knots[4]+4.
    t[1:-1]+=0.25
    fresh=model(m.knots.copy(),m.coefficients.copy())
    np.testing.assert_array_equal(m.center(t),fresh.center(t))
    np.testing.assert_array_equal(m.curvature(t),fresh.curvature(t))
