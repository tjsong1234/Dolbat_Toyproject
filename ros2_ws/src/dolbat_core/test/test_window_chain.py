from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import yaml
from dolbat_core.window_chain_detection import WindowChainDetector,observed_error
from dolbat_core.geometry_preprocessing import GeometryPreprocessor,retain_road_adjacent_components
from dolbat_core.preprocessing import Preprocessor
from dolbat_core.lane_detection import LaneFit

ROOT=Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT/'dolbat_core/resources/config/geometry.yaml').read_text(encoding='utf-8'))


def detector():
    g=config();return WindowChainDetector(g['lane_detection'],g['lane_validation'])


def pair():
    mask=np.zeros((480,640),np.uint8)
    for y in (20,180,340):cv2.rectangle(mask,(134,y),(146,y+80),255,-1)
    cv2.rectangle(mask,(494,0),(506,479),255,-1)
    return mask


def test_window_chain_connects_dashes_and_keeps_fresh_observations():
    d=detector();r=d.detect(pair())
    assert r.status=='DETECTED'
    for fit,x in ((r.left,140),(r.right,500)):
        assert not fit.inferred and abs(fit.x(250)-x)<2
        assert all(len(node.rows)>=6 for node in fit.chain)
    blank=d.detect(np.zeros((480,640),np.uint8))
    assert not blank.usable and blank.left is None and blank.right is None


def test_single_dashed_observation_is_left_of_recorded_target_lane():
    mask=pair();mask[:,400:]=0;r=detector().detect(mask)
    assert r.status=='INFERRED_RIGHT' and r.left.dashed
    assert not r.left.inferred and r.right.inferred and len(r.right.pixels)==0


def test_temporal_comparison_never_extrapolates_outside_shared_observation():
    a=LaneFit(np.array([.002,-1,300]),np.empty((0,2)),1,.4,200,400)
    b=LaneFit(np.array([0,0,170]),np.empty((0,2)),1,.2,0,100)
    assert observed_error(a,b) is None


def test_smoothing_cannot_move_fresh_curve_more_than_configured_bound():
    d=detector();d.detect(pair());shifted=np.roll(pair(),10,axis=1);r=d.detect(shifted)
    assert r.status=='DETECTED'
    for f in (r.left,r.right):
        y=np.unique(f.pixels[:,1])
        assert np.max(abs(f.x(y)-np.polyval(f.raw_coefficients,y)))<=2.5+1e-8
        assert np.median(abs(f.pixels[:,0]-f.x(f.pixels[:,1])))<5


def test_expanded_roi_is_independent_of_homography_anchor_quad():
    base=yaml.safe_load((ROOT/'dolbat_core/resources/config/settings.yaml').read_text(encoding='utf-8'))['preprocessing']
    marking=yaml.safe_load((ROOT/'dolbat_core/resources/config/perception_improved.yaml').read_text(encoding='utf-8'))['marking_suppression']
    frame=np.full((480,640,3),70,np.uint8);frame[345:356,45:56]=245
    new=GeometryPreprocessor(base,config(),marking);a=new.process(frame)
    old=Preprocessor(base).process(frame)
    assert old.roi[350,50].max()==0 and a.roi[350,50].min()==245
    pt=cv2.perspectiveTransform(np.float32([[[50,350]]]),new.matrix)[0,0]
    assert 0<pt[0]<new.size[0] and 0<pt[1]<new.size[1]
    anchors=cv2.perspectiveTransform(new.anchor_points.astype(np.float32).reshape(-1,1,2),new.matrix).reshape(-1,2)
    expected=np.array(config()['preprocessing']['destination_points'])*(np.array(new.size)-1)
    np.testing.assert_allclose(anchors,expected,atol=.001)


def test_road_adjacency_preserves_the_entire_center_of_wide_white_lane():
    paint=np.zeros((80,100),np.uint8);paint[:,40:65]=255
    road=np.zeros_like(paint);road[:,:45]=255
    kept=retain_road_adjacent_components(paint,road)
    np.testing.assert_array_equal(kept,paint)
    assert kept[:,50:60].all()


def test_solid_line_is_not_classified_as_dashed_due_to_preprocessing_holes():
    raw=pair();binary=raw.copy();binary[180:260,494:507]=0
    r=detector().detect(binary,raw)
    assert r.status=='DETECTED'
    assert r.left.dashed and not r.right.dashed
