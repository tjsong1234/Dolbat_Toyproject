from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import yaml
from dolbat_core.supported_lane_detection import SupportedLaneDetector

ROOT=Path(__file__).resolve().parents[1]


def detectors():
    cfg=yaml.safe_load((ROOT/'dolbat_core/resources/config/settings.yaml').read_text(encoding='utf-8'))['lane_detection']
    validation=yaml.safe_load((ROOT/'dolbat_core/resources/config/perception_improved.yaml').read_text(encoding='utf-8'))['lane_validation']
    return SupportedLaneDetector(deepcopy(cfg),validation,trace=True),SupportedLaneDetector(deepcopy(cfg),validation)


def mask_pair():
    mask=np.zeros((480,640),np.uint8)
    for y in (0,140,280,420):cv2.rectangle(mask,(114,y),(126,min(479,y+75)),255,-1)
    cv2.rectangle(mask,(479,0),(491,479),255,-1)
    return mask


def test_trace_records_actual_twelve_disjoint_windows_and_observed_pixels():
    d,_=detectors();mask=mask_pair();r=d.detect(mask)
    assert r.status=='DETECTED'
    for side in ('left','right'):
        t=d.trace['selected'][side];assert len(t['steps'])==12
        all_rows=[]
        for step in t['steps']:
            x1,lo,x2,hi=step['bounds'];all_rows.extend(range(lo,hi))
            px=step['pixels'].astype(int)
            assert ((px[:,1]>=lo)&(px[:,1]<hi)).all()
            assert (mask[px[:,1],px[:,0]]!=0).all()
            assert np.all(abs(px[:,0]-step['center_before'])<=55)
            if step['median'] is not None:
                assert step['center_next']==step['median']+step['slope']
        assert sorted(all_rows)==list(range(480))


def test_optional_tracing_preserves_temporal_results_and_empty_is_not_observed():
    d,plain=detectors();pair=mask_pair();one=pair.copy();one[:,400:]=0
    for mask in (pair,pair,one,np.zeros_like(pair),pair):
        a,b=d.detect(mask),plain.detect(mask)
        assert a.status==b.status
        for left,right in ((a.left,b.left),(a.right,b.right)):
            if left is None:assert right is None
            else:np.testing.assert_array_equal(left.coefficients,right.coefficients)
    blank=d.detect(np.zeros_like(pair))
    assert not blank.usable and blank.left is None and blank.right is None


def test_inferred_boundary_has_no_observation_trace():
    d,_=detectors();mask=mask_pair();mask[:,400:]=0
    r=d.detect(mask)
    assert r.status=='INFERRED_RIGHT'
    assert d.trace['selected']['left'] is not None
    assert d.trace['selected']['right'] is None
    assert r.right.inferred and not len(r.right.pixels)


def test_nonvertical_curve_is_fit_as_x_of_y():
    d,_=detectors();mask=np.zeros((480,640),np.uint8)
    yy=np.arange(480);xx=120+.00025*(yy-240)**2
    for offset in (0,365):
        points=np.column_stack((xx+offset,yy)).astype(np.int32)
        cv2.polylines(mask,[points],False,255,13)
    r=d.detect(mask)
    assert r.status=='DETECTED'
    for y in (100,240,400):
        assert abs(r.left.x(y)-(120+.00025*(y-240)**2))<3
