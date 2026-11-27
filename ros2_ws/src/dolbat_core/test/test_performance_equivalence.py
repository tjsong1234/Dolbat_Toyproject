"""Exact mask preservation checks for CPU-only implementation changes."""
import cv2
import numpy as np
from dolbat_core.marking_suppression import short_vertical_runs,suppress_markings
from dolbat_core.geometry_preprocessing import retain_road_adjacent_components
from dolbat_core.road_surface_preprocessing import RoadSurfacePreprocessor
from dolbat_core.geometry_preprocessing import GeometryPreprocessor


def column_reference(paint,limit):
    expected=np.zeros_like(paint)
    for x in range(paint.shape[1]):
        start=None
        for y in range(paint.shape[0]+1):
            on=y<paint.shape[0] and paint[y,x]!=0
            if on and start is None:start=y
            if not on and start is not None:
                if y-start<=limit:expected[start:y,x]=255
                start=None
    return expected


def test_run_threshold_preserves_finite_image_borders_even_and_odd_kernels():
    rng=np.random.default_rng(917)
    for shape in ((1,1),(3,5),(40,31),(65,81)):
        for limit in (0,1,2,3,24,25,26,28,100):
            for j in range(5):
                paint=np.uint8(rng.random(shape)<(.15+j*.18))*255
                if j==4:paint[:,0]=255
                np.testing.assert_array_equal(short_vertical_runs(paint,limit),column_reference(paint,limit))


def test_road_contact_counts_unchanged_when_background_is_skipped():
    rng=np.random.default_rng(112)
    for fraction in (.01,.12,.7):
        candidate=np.uint8(rng.random((60,80))>.67)*255
        candidate[15:45,5:10]=255
        road=np.uint8(rng.random(candidate.shape)>.5)*255
        n,labels,stats,_=cv2.connectedComponentsWithStats(candidate)
        total=np.bincount(labels.ravel(),minlength=n)
        contact=np.bincount(labels[road!=0],minlength=n)
        keep=(contact>=np.maximum(1,total*fraction))&(stats[:,cv2.CC_STAT_AREA]>=5)&(stats[:,cv2.CC_STAT_HEIGHT]>=8)
        keep[0]=False
        np.testing.assert_array_equal(retain_road_adjacent_components(candidate,road,5,fraction),np.uint8(keep[labels])*255)


def test_reused_hsv_does_not_change_any_suppression_output():
    bev=np.zeros((100,130,3),np.uint8)
    bev[:,20:27]=245;bev[40:46,10:120]=255
    bev[10:30,70:79]=255;bev[10:30,90:99]=255
    candidate=cv2.inRange(bev,(200,200,200),(255,255,255))
    config={'mode':'thickness_local','horizontal_length_px':30,'horizontal_max_vertical_thickness_px':7}
    original=suppress_markings(bev,candidate,config)
    reused=suppress_markings(bev,candidate,config,hsv=cv2.cvtColor(bev,cv2.COLOR_BGR2HSV))
    for name in ('binary','raw_paint','crosswalk','horizontal','removed'):
        np.testing.assert_array_equal(getattr(original,name),getattr(reused,name))
    assert original.stripe_count==reused.stripe_count


def test_stricter_road_contact_gate_dominates_broad_gate():
    rng=np.random.default_rng(761)
    for probability in (.15,.5,.85):
        for fraction in (.01,.12,.7):
            candidate=np.uint8(rng.random((60,80))<probability)*255
            candidate[5:50,10:20]=255
            broad=np.uint8(rng.random(candidate.shape)<.7)*255
            narrow=broad & (np.uint8(rng.random(candidate.shape)<.6)*255)
            first=retain_road_adjacent_components(candidate,broad,5,fraction)
            both=retain_road_adjacent_components(first,narrow,5,fraction)
            strict_only=retain_road_adjacent_components(candidate,narrow,5,fraction)
            np.testing.assert_array_equal(both,strict_only)


def test_road_gate_elision_is_limited_to_proven_white_mode():
    obj=object.__new__(RoadSurfacePreprocessor)
    obj.cfg={'mask_mode':'white_thickness','road_max_saturation':90,'surface_max_saturation':70}
    assert not obj.requires_initial_road_gate()
    obj.cfg['surface_max_saturation']=100
    assert obj.requires_initial_road_gate()
    obj.cfg['surface_max_saturation']=70;obj.cfg['mask_mode']='refined'
    assert obj.requires_initial_road_gate()
    assert object.__new__(GeometryPreprocessor).requires_initial_road_gate()
