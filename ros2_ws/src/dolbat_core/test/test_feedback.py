from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import yaml
from dolbat_core.marking_suppression import suppress_markings,vertical_run_lengths
from dolbat_core.feedback_lane_detection import FeedbackLaneDetector
from dolbat_core.supported_lane_detection import SupportedLaneDetector

ROOT=Path(__file__).resolve().parents[1]


def config():
    cfg=yaml.safe_load((ROOT/'dolbat_core/resources/config/perception_improved.yaml').read_text(encoding='utf-8'))['marking_suppression']
    cfg.update(yaml.safe_load((ROOT/'dolbat_core/resources/config/feedback.yaml').read_text(encoding='utf-8'))['marking_overrides'])
    return cfg


def image_with_stop_line():
    im=np.full((480,640,3),70,np.uint8)
    cv2.rectangle(im,(100,0),(112,479),(245,245,245),-1)
    cv2.rectangle(im,(20,350),(620,361),(245,245,245),-1)
    return im


def test_directional_thickness_is_exact_at_image_edges_and_gaps():
    b=np.zeros((8,4),np.uint8);b[:3,0]=255;b[4:,0]=255;b[:,1]=255;b[3,2]=255
    runs=vertical_run_lengths(b)
    np.testing.assert_array_equal(runs[:,0],[3,3,3,0,4,4,4,4])
    assert np.all(runs[:,1]==8) and runs[3,2]==1 and not runs[:,3].any()


def test_crossing_lane_is_preserved_even_at_stop_line_intersection():
    im=image_with_stop_line();candidate=np.uint8(im[:,:,0]>200)*255
    r=suppress_markings(im,candidate,config())
    np.testing.assert_array_equal(r.binary[:,100:113],candidate[:,100:113])
    assert not r.binary[350:362,150:600].any()
    assert not np.any(r.removed & cv2.bitwise_not(r.raw_paint))


def test_nearby_disconnected_dash_is_not_cut_by_padding_or_extrapolation():
    im=np.full((480,640,3),70,np.uint8)
    cv2.rectangle(im,(120,220),(480,230),(245,245,245),-1)
    cv2.rectangle(im,(505,190),(518,260),(245,245,245),-1)
    candidate=np.uint8(im[:,:,0]>200)*255
    r=suppress_markings(im,candidate,config())
    np.testing.assert_array_equal(r.binary[:,505:519],candidate[:,505:519])
    assert not r.binary[220:231,120:481].any()


def test_actual_slanted_horizontal_paint_removed_without_touching_neighbor_lane():
    im=np.full((480,640,3),70,np.uint8)
    cv2.line(im,(-15,212),(620,252),(245,245,245),14)
    cv2.rectangle(im,(580,0),(592,479),(245,245,245),-1)
    candidate=np.uint8(im[:,:,0]>200)*255
    r=suppress_markings(im,candidate,config())
    assert not r.binary[200:230,:40].any()
    np.testing.assert_array_equal(r.binary[:,580:593],candidate[:,580:593])


def test_local_masks_never_delete_outside_observed_paint_randomized():
    im=np.random.default_rng(15).integers(0,256,(100,200,3),dtype=np.uint8)
    candidate=np.full((100,200),255,np.uint8)
    r=suppress_markings(im,candidate,config())
    assert not np.any((r.crosswalk|r.horizontal)&cv2.bitwise_not(r.raw_paint))


def detector(trace=False):
    cfg=yaml.safe_load((ROOT/'dolbat_core/resources/config/settings.yaml').read_text(encoding='utf-8'))['lane_detection']
    val=yaml.safe_load((ROOT/'dolbat_core/resources/config/perception_improved.yaml').read_text(encoding='utf-8'))['lane_validation']
    cfg['margin_px']=75
    return FeedbackLaneDetector(cfg,val,115,trace)


def test_retry_does_not_double_count_missing_frames_or_show_stale_observations():
    d=detector(True);mask=np.zeros((480,640),np.uint8)
    cv2.rectangle(mask,(114,0),(126,479),255,-1)
    assert d.detect(mask).usable
    for missed in range(1,6):
        r=d.detect(np.zeros_like(mask))
        assert not r.usable and r.left is None and r.right is None
        assert d.missed==missed and r.retry_attempted and not r.retry_accepted
    assert d.previous is None and d.retries==5 and d.retry_successes==0


def test_valid_primary_observations_are_not_replaced_by_a_wider_search():
    d=detector();mask=np.zeros((480,640),np.uint8)
    cv2.rectangle(mask,(114,0),(126,479),255,-1)
    cv2.rectangle(mask,(479,0),(491,479),255,-1)
    r=d.detect(mask)
    assert r.status=='DETECTED' and not r.retry_attempted and r.search_margin_px==75
    assert d.retries==0 and d.cfg['margin_px']==75


def test_wide_retry_reacquires_two_separated_dashes_without_inventing_partner_pixels():
    cfg=yaml.safe_load((ROOT/'dolbat_core/resources/config/settings.yaml').read_text(encoding='utf-8'))['lane_detection']
    val=yaml.safe_load((ROOT/'dolbat_core/resources/config/perception_improved.yaml').read_text(encoding='utf-8'))['lane_validation']
    mask=np.zeros((480,640),np.uint8)
    cv2.rectangle(mask,(114,420),(126,479),255,-1)
    cv2.rectangle(mask,(214,240),(226,319),255,-1)
    assert not SupportedLaneDetector(cfg,val).detect(mask).usable
    d=FeedbackLaneDetector(cfg,val,115);r=d.detect(mask)
    assert r.status=='INFERRED_RIGHT' and r.retry_accepted and r.search_margin_px==115
    assert r.left.support['rows']==140 and not r.left.inferred
    assert r.right.inferred and len(r.right.pixels)==0
    assert d.missed==0 and d.retry_successes==1
