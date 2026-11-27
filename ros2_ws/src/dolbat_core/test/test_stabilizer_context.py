"""Reset semantics depend on current geometric evidence, not UI status names."""
from pathlib import Path
import numpy as np
import pytest
import yaml
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.metric_planning import MetricPlan,MetricBoundary
from dolbat_core.stable_metric_planning import StableMetricLanePlanner

ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/'dolbat_core/resources/config/week3.yaml').read_text(encoding='utf-8'))['planning']
MAPPING=MetricMapping(.25,.5,319.5,639)


def planner():
    return StableMetricLanePlanner(MAPPING,{**CONFIG,'path_smoothing_time_constant_s':.075,
        'max_path_adjustment_cm':1.,'path_pair_range_reset_cm':20.},90,fps=24)


def raw(label='arbitrary display label',offset=0.,lo=20.,hi=240.,left=None,right=None):
    xy=np.column_stack((np.linspace(lo,hi,160),np.full(160,offset)))
    return MetricPlan(label,'current frame',left,right,xy,MAPPING.cm_to_pixels(xy),
        None,None,np.zeros(160),'test')


def context(side='right',both=True,interval=(20.,240.)):
    return {'reference_side':side,'both_observed':both,'observation_range_cm':interval}


def boundary(interval=(20.,240.)):
    x=np.linspace(*interval,20);xy=np.column_stack((x,np.zeros(20)))
    return MetricBoundary(np.zeros(3),xy,xy,np.ones(20,bool),interval,0.,0.)


def test_reset_and_blending_are_invariant_to_display_status_names():
    a,b=planner(),planner()
    for i,(near,far) in enumerate([(20.,240.),(20.,240.),(30.,280.),(30.,280.)]):
        r1=a.stabilize(raw('OBSERVED_PAIR',offset=i*5,lo=near,hi=far),context=context(interval=(near,far)))
        r2=b.stabilize(raw(['anything','ESTIMATED_FROM_RIGHT_WITH_LEFT_SUPPORT','renamed','OBSERVED_PAIR'][i],
            offset=i*5,lo=near,hi=far),context=context(interval=(near,far)))
        np.testing.assert_allclose(r1.path_cm,r2.path_cm,atol=0.)
        assert r1.stabilization_reset_reason==r2.stabilization_reset_reason
        assert r1.stabilization_applied==r2.stabilization_applied
    assert r2.stabilization_applied


def test_actual_pair_range_change_resets_new_yellow_guided_status():
    p=planner();label='ESTIMATED_FROM_RIGHT_WITH_LEFT_SUPPORT'
    p.stabilize(raw(label),context=context())
    result=p.stabilize(raw(label,offset=10.,hi=270.),context=context(interval=(20.,270.)))
    assert result.stabilization_reset_reason=='paired_observation_range_changed'
    assert not result.stabilization_applied and result.stabilization_range_jump_cm==pytest.approx(30.)
    np.testing.assert_array_equal(result.path_cm,result.raw_path_cm)


def test_changed_observation_support_resets_even_when_completed_path_range_is_constant():
    p=planner();p.stabilize(raw(),context=context(interval=(80.,150.)))
    result=p.stabilize(raw(offset=8.),context=context(interval=(110.,150.)))
    assert result.stabilization_reset_reason=='paired_observation_range_changed'
    assert result.stabilization_range_jump_cm==pytest.approx(30.)
    assert not result.stabilization_applied


def test_raw_support_range_still_resets_when_observation_interval_is_constant():
    p=planner();p.stabilize(raw(),context=context(interval=(80.,150.)))
    result=p.stabilize(raw(offset=8.,hi=265.),context=context(interval=(80.,150.)))
    assert result.stabilization_range_jump_cm==pytest.approx(25.)
    assert result.stabilization_reset_reason=='paired_observation_range_changed'


def test_one_sided_range_change_keeps_current_overlap_only_filter():
    p=planner();p.stabilize(raw(),context=context(both=False))
    current=raw(offset=10.,hi=280.)
    result=p.stabilize(current,context=context(both=False,interval=(20.,280.)))
    assert result.stabilization_applied and result.max_path_adjustment_cm<=1.+1e-10
    new=result.path_cm[:,0]>240.
    np.testing.assert_array_equal(result.path_cm[new],current.path_cm[new])


def test_reference_flip_resets_with_identical_status_labels():
    p=planner();p.stabilize(raw('same'),context=context(side='left',both=False))
    result=p.stabilize(raw('same',offset=10.),context=context(side='right',both=False))
    assert result.stabilization_reset_reason=='observed_side_changed'
    assert not result.stabilization_applied


def test_legacy_callers_derive_context_from_current_boundary_objects():
    p=planner();left,right=boundary(),boundary()
    first=p.stabilize(raw('display name',left=left,right=right))
    assert first.stabilization_context==context()
    moved_right=boundary((20.,270.));moved_left=boundary((20.,270.))
    result=p.stabilize(raw('different name',offset=10.,left=moved_left,right=moved_right))
    assert result.stabilization_reset_reason=='paired_observation_range_changed'


def test_context_snapshots_do_not_alias_caller_or_planner_history():
    p=planner();evidence=context(interval=[20.,240.])
    first=p.stabilize(raw(),context=evidence)
    evidence['reference_side']='left';evidence['observation_range_cm'][0]=99.
    assert first.stabilization_context==context()
    second=p.stabilize(raw(offset=5.),context=context())
    assert second.stabilization_previous_context==context()
    second.stabilization_context['reference_side']='left'
    assert p.previous_context['reference_side']=='right'


def test_missing_path_clears_explicit_context_and_cannot_replay_it():
    p=planner();p.stabilize(raw(),context=context())
    empty=np.empty((0,2));blank=MetricPlan('blank','no current geometry',None,None,
        empty,empty,None,None,np.array([]),'none')
    result=p.stabilize(blank,context=context(side='none',both=False,interval=None))
    assert not result.usable and p.previous_context is None and p.previous_path_cm is None
    recovered=p.stabilize(raw(offset=20.),context=context(side='left',both=False))
    assert not recovered.stabilization_applied
    assert recovered.stabilization_reset_reason=='first_current_path'


@pytest.mark.parametrize('bad',[{'reference_side':'typo'},
    {'observation_range_cm':(240.,20.)},{'observation_range_cm':(0.,np.nan)},
    {'observation_range_cm':(0.,20.,30.)}])
def test_invalid_geometric_context_is_rejected(bad):
    with pytest.raises(ValueError):planner().stabilize(raw(),context=bad)
