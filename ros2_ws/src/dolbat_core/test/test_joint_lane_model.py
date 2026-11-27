import numpy as np
import pytest
from dolbat_core.bspline_basis import clamped_knots,bspline_basis
from dolbat_core.joint_lane_model import fit_joint_lane


DOMAIN=(20.,260.)


def road(t,width=90.,f=lambda x:.001*x*x+.1*x+4,m=lambda x:.002*x+.1):
    y=f(t);slope=m(t);q=np.sqrt(1+slope*slope)
    normal=np.column_stack((-slope/q,1/q));center=np.column_stack((t,y))
    return center,center+width/2*normal,center-width/2*normal


def fit(left,right,initial=None,config=None,partial_right=False):
    t=np.linspace(*DOMAIN,120)
    if initial is None:initial=road(t)[0]+np.array([0,3.])
    evidence=[]
    if left is not None:evidence.append({'side':'left','points_cm':left,'confidence':1.})
    if right is not None:evidence.append({'side':'right','points_cm':right,'confidence':1.,'partial':partial_right})
    return fit_joint_lane(evidence,DOMAIN,initial,90.,config or {})


def test_clamped_basis_partition_and_endpoint_interpolation():
    knots=clamped_knots(DOMAIN,7);x=np.linspace(*DOMAIN,301);b=bspline_basis(x,knots)
    np.testing.assert_allclose(b.sum(axis=1),1,atol=1e-12)
    np.testing.assert_allclose(b[0],[1,0,0,0,0,0,0],atol=1e-12)
    np.testing.assert_allclose(b[-1],[0,0,0,0,0,0,1],atol=1e-12)
    assert b.min()>=0


@pytest.mark.parametrize('order',[1,2,3])
def test_analytic_derivatives_match_finite_difference_away_from_knots(order):
    knots=clamped_knots(DOMAIN,7);x=np.array([31.,63.,111.,169.,237.]);eps=1e-3
    numeric=(bspline_basis(x+eps,knots,derivative=order-1)-bspline_basis(x-eps,knots,derivative=order-1))/(2*eps)
    np.testing.assert_allclose(bspline_basis(x,knots,derivative=order),numeric,rtol=1e-6,atol=1e-10)
    np.testing.assert_allclose(bspline_basis(x,knots,derivative=order).sum(axis=1),0,atol=1e-12)


def test_straight_shared_model_recovers_center_and_width_not_weighted_boundary_average():
    t=np.linspace(*DOMAIN,200);center=np.column_stack((t,np.full_like(t,12.)))
    left=center+[0,47.];right=center-[0,47.]
    model=fit(left,right,initial=center+[0,7])
    assert model.accepted,model.reason
    np.testing.assert_allclose(model.center(t)[:,1],12,atol=.03)
    np.testing.assert_allclose(model.width(t),94,atol=.08)
    assert max(abs(model.curvature(t)))<1e-5


def test_curved_real_normal_pair_jointly_recovers_geometric_center():
    t=np.linspace(*DOMAIN,200);center,left,right=road(t)
    model=fit(left,right,config={'joint_curvature_regularization':1e-6,'joint_jerk_regularization':1e-7})
    assert model.accepted,model.reason
    np.testing.assert_allclose(model.center(t),center,atol=.15)
    assert max(abs(model.width(t)-90))<.25
    np.testing.assert_allclose((model.boundary('left',t)+model.boundary('right',t))/2,model.center(t),atol=1e-12)
    assert model.diagnostics['generated_observation_rows']==0


def test_single_current_boundary_uses_width_prior_and_does_not_invent_second_evidence():
    t=np.linspace(*DOMAIN,180);center,left,right=road(t)
    model=fit(None,right,config={'joint_curvature_regularization':1e-6,'joint_jerk_regularization':1e-7})
    assert model.accepted,model.reason
    np.testing.assert_allclose(model.width(t),90,atol=.1)
    np.testing.assert_allclose(model.center(t),center,atol=.4)
    assert model.diagnostics['full_sides']==['right']
    assert len(model.evidence_weights)==1


def test_dash_gaps_are_modelled_without_adding_observed_rows():
    t=np.linspace(*DOMAIN,241);center,left,right=road(t)
    keep=((t>=35)&(t<80))|((t>=130)&(t<170))|((t>=220)&(t<250))
    model=fit(left[keep],right)
    assert model.accepted,model.reason
    assert len(model.center(t))==len(t)
    assert model.diagnostics['actual_selected_rows']==int(keep.sum())+240
    assert model.evidence_weights[0]['used_rows']<=keep.sum()
    assert model.diagnostics['max_abs_center_curvature_per_cm']<.004


def test_isolated_outliers_cannot_create_local_curvature_spike():
    t=np.linspace(*DOMAIN,180);center,left,right=road(t)
    left[70:75,1]+=25
    model=fit(left,right)
    assert model.accepted,model.reason
    assert max(abs(model.curvature(t)))<.006
    assert np.median(np.linalg.norm(model.center(t)-center,axis=1))<.6
    assert min(model.evidence_weights[0]['weights'])<np.median(model.evidence_weights[0]['weights'])


def test_wrong_side_or_crossed_observations_are_not_accepted_as_valid_lane():
    t=np.linspace(*DOMAIN,160);center,left,right=road(t)
    bad=fit(right,left)
    assert not bad.accepted
    assert bad.reason in ('insufficient_current_full_boundary_agreement','joint_residual_too_large','joint_boundary_forward_fold','joint_curvature_guard')


def test_partial_fragment_stays_partial_and_has_lower_total_precision():
    t=np.linspace(*DOMAIN,180);center,left,right=road(t)
    model=fit(left,right[75:100],partial_right=True)
    assert model.accepted,model.reason
    assert model.diagnostics['full_sides']==['left']
    assert model.evidence_weights[1]['partial']
    assert sum(model.evidence_weights[1]['weights'])<sum(model.evidence_weights[0]['weights'])
    assert model.diagnostics['partial_selected_rows']==25


def test_no_current_full_observation_is_rejected():
    t=np.linspace(*DOMAIN,60);_,left,right=road(t)
    initial=road(t)[0]
    model=fit_joint_lane([{'side':'left','points_cm':left,'partial':True}],DOMAIN,initial,90,{})
    assert not model.accepted and model.reason=='no_current_full_boundary_evidence'


def test_model_refuses_implicit_extrapolation_outside_its_evidence_domain():
    t=np.linspace(*DOMAIN,100);_,left,right=road(t);model=fit(left,right)
    with pytest.raises(ValueError,match='outside'):
        model.center([DOMAIN[1]+1])
