import numpy as np
import pytest
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.week4_geometry import rear_reference


def test_original_bottom_not_bev_bottom_and_lateral_translation():
    mapping=MetricMapping(1.,2.,50.,100.)
    h=np.array([[1.,0.,10.],[0.,1.,20.],[0.,0.,1.]])
    result=rear_reference(h,mapping,(101,51),.9)
    assert result['source_bottom_bev_px']==[60.,70.]
    assert result['source_bottom_bev_cm']==[60.,-10.]
    np.testing.assert_allclose(result['rear_translation_m'],[.3,.1])
    anchor=np.array(result['source_bottom_bev_cm'])/100+result['rear_translation_m']
    np.testing.assert_allclose(anchor,[.9,0.])


def test_projective_anchor_and_metre_conversion():
    h=np.array([[1.,0.,0.],[0.,1.,0.],[0.,.01,1.]])
    mapping=MetricMapping(.2,.5,50.,100.)
    result=rear_reference(h,mapping,(101,101),.9)
    np.testing.assert_allclose(result['source_bottom_bev_px'],[25.,50.])
    np.testing.assert_allclose(result['rear_translation_m'],[.65,-.05])


def test_anchor_at_infinity_is_rejected():
    h=np.array([[1.,0.,0.],[0.,1.,0.],[0.,1.,-100.]])
    with pytest.raises(ValueError,match='infinity'):
        rear_reference(h,MetricMapping(1,1,0,0),(101,101))
