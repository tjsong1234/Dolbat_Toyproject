from pathlib import Path
import cv2
import numpy as np
import yaml
from dolbat_core.marking_suppression import suppress_markings
from dolbat_core.supported_lane_detection import fit_supported_rows,SupportedLaneDetector


ROOT=Path(__file__).resolve().parents[1]


def settings():
    cfg=yaml.safe_load((ROOT/'dolbat_core/resources/config/settings.yaml').read_text(encoding='utf-8'))
    profile=yaml.safe_load((ROOT/'dolbat_core/resources/config/perception_improved.yaml').read_text(encoding='utf-8'))
    return cfg['lane_detection'],profile


def test_crosswalk_removed_without_erasing_neighboring_lane():
    _,profile=settings()
    image=np.full((480,640,3),70,np.uint8)
    for x in (170,275,380,485):
        cv2.rectangle(image,(x,140),(x+45,370),(245,245,245),-1)
    cv2.rectangle(image,(100,0),(112,479),(245,245,245),-1)
    cv2.rectangle(image,(600,0),(612,479),(245,245,245),-1)
    candidate=np.uint8(image[:,:,0]>200)*255
    result=suppress_markings(image,candidate,profile['marking_suppression'])
    assert result.stripe_count==4
    assert not result.binary[140:371,170:531].any()
    assert np.array_equal(result.binary[:,100:113],candidate[:,100:113])
    assert np.array_equal(result.binary[:,600:613],candidate[:,600:613])


def test_two_lane_dashes_do_not_form_a_crosswalk_group():
    _,profile=settings()
    image=np.full((480,640,3),70,np.uint8)
    for x in (100,500):
        cv2.rectangle(image,(x,100),(x+30,230),(245,245,245),-1)
    candidate=np.uint8(image[:,:,0]>200)*255
    result=suppress_markings(image,candidate,profile['marking_suppression'])
    assert result.stripe_count==0
    assert np.array_equal(result.binary,candidate)


def test_stop_line_is_suppressed_with_most_of_vertical_lane_preserved():
    _,profile=settings()
    image=np.full((480,640,3),70,np.uint8)
    cv2.rectangle(image,(100,0),(112,479),(245,245,245),-1)
    cv2.rectangle(image,(10,350),(620,361),(245,245,245),-1)
    candidate=np.uint8(image[:,:,0]>200)*255
    result=suppress_markings(image,candidate,profile['marking_suppression'])
    assert not result.binary[350:362,150:600].any()
    assert np.count_nonzero(result.binary[:,100:113])>.9*np.count_nonzero(candidate[:,100:113])


def test_sparse_top_and_bottom_pixels_cannot_create_observed_lane():
    cfg,profile=settings()
    mask=np.zeros((480,640),np.uint8)
    cv2.rectangle(mask,(114,30),(126,90),255,-1)
    cv2.rectangle(mask,(10,450),(630,454),255,-1)
    for y in (150,290,380):
        cv2.rectangle(mask,(115,y),(125,y+2),255,-1)
    y,x=mask.nonzero()
    fit,reason,details=fit_supported_rows(x,y,cfg,profile['lane_validation'],480)
    assert fit is None and reason!='supported'
    assert details['rows']<80


def test_row_balance_resists_horizontal_line_and_preserves_dashes():
    cfg,profile=settings()
    mask=np.zeros((480,640),np.uint8)
    for y0 in (0,140,280,420):
        cv2.rectangle(mask,(114,y0),(126,min(479,y0+75)),255,-1)
    cv2.rectangle(mask,(0,355),(639,360),255,-1)
    y,x=mask.nonzero()
    fit,reason,details=fit_supported_rows(x,y,cfg,profile['lane_validation'],480)
    assert fit is not None,reason
    assert abs(fit.x(350)-120)<2
    assert details['support_fraction']>.4


def test_supported_detector_never_calls_inferred_partner_observed():
    cfg,profile=settings()
    mask=np.zeros((480,640),np.uint8)
    cv2.rectangle(mask,(114,0),(126,479),255,-1)
    result=SupportedLaneDetector(cfg,profile['lane_validation']).detect(mask)
    assert result.status=='INFERRED_RIGHT'
    assert not result.left.inferred and result.right.inferred


def test_tiny_far_fragment_does_not_invalidate_two_real_dashes():
    cfg,profile=settings()
    mask=np.zeros((480,640),np.uint8)
    for y0,y1 in ((10,11),(260,350),(420,470)):
        cv2.rectangle(mask,(114,y0),(126,y1),255,-1)
    y,x=mask.nonzero()
    fit,reason,details=fit_supported_rows(x,y,cfg,profile['lane_validation'],480)
    assert fit is not None,reason
    assert fit.y_min==260 and fit.y_max==470
    assert abs(fit.x(400)-120)<2


def test_optimized_raw_paint_threshold_matches_channel_definition():
    _,profile=settings()
    image=np.random.default_rng(42).integers(0,256,(80,100,3),dtype=np.uint8)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    expected=np.uint8((image.min(axis=2)>=175)&(hsv[:,:,1]<=65))*255
    result=suppress_markings(image,np.zeros((80,100),np.uint8),profile['marking_suppression'])
    np.testing.assert_array_equal(result.raw_paint,expected)


def test_clipped_dash_and_long_dash_keep_valid_observation_support():
    cfg,profile=settings()
    mask=np.zeros((480,640),np.uint8)
    for y0,y1 in ((107,117),(262,373)):
        cv2.rectangle(mask,(114,y0),(126,y1),255,-1)
    y,x=mask.nonzero()
    fit,reason,details=fit_supported_rows(x,y,cfg,profile['lane_validation'],480)
    assert fit is not None,reason
    assert details['max_gap_px']==144


def test_slanted_stop_line_end_is_removed_without_erasing_lane_body():
    _,profile=settings()
    image=np.full((480,640,3),70,np.uint8)
    cv2.line(image,(-15,212),(620,252),(245,245,245),14)
    cv2.rectangle(image,(580,0),(592,479),(245,245,245),-1)
    candidate=np.uint8(image[:,:,0]>200)*255
    result=suppress_markings(image,candidate,profile['marking_suppression'])
    assert not result.binary[200:230,:40].any()
    assert np.count_nonzero(result.binary[:,580:593])>.9*np.count_nonzero(candidate[:,580:593])


def test_row_reduction_is_independent_of_pixel_order():
    cfg,profile=settings()
    mask=np.zeros((480,640),np.uint8)
    for start in (0,140,280,420):
        cv2.rectangle(mask,(114,start),(126,min(479,start+75)),255,-1)
    y,x=mask.nonzero();order=np.random.default_rng(7).permutation(len(x))
    a,_,_=fit_supported_rows(x,y,cfg,profile['lane_validation'],480)
    b,_,_=fit_supported_rows(x[order],y[order],cfg,profile['lane_validation'],480)
    assert a is not None and b is not None
    np.testing.assert_allclose(a.coefficients,b.coefficients,atol=1e-10)
