"""Regressions for a horizontal seed accidentally erasing a bending lane."""
from pathlib import Path
import cv2
import numpy as np
import yaml
from dolbat_core.marking_suppression import suppress_markings

ROOT=Path(__file__).resolve().parents[1]


def config():
    base=yaml.safe_load((ROOT/'dolbat_core/resources/config/perception_improved.yaml').read_text(encoding='utf-8'))
    geometry=yaml.safe_load((ROOT/'dolbat_core/resources/config/geometry.yaml').read_text(encoding='utf-8'))
    return {**base['marking_suppression'],**geometry['marking_overrides']}


def test_horizontal_seed_cannot_erase_connected_curved_lane():
    im=np.full((640,640,3),70,np.uint8)
    x=np.arange(480);y=40+.16*x+.0012*x*x
    points=np.column_stack((x,y)).astype(np.int32)
    cv2.polylines(im,[points],False,(245,245,245),16)
    paint=np.uint8(im[:,:,0]>200)*255
    cfg=config();old={k:v for k,v in cfg.items() if k not in (
        'horizontal_max_center_slope','horizontal_max_center_rms_px','horizontal_min_component_span_px')}
    before=suppress_markings(im,paint,old)
    assert np.count_nonzero(before.removed)>1000, 'Fixture must reproduce the original failure'
    after=suppress_markings(im,paint,cfg)
    np.testing.assert_array_equal(after.binary,paint)


def test_straight_and_slanted_stop_lines_still_removed_with_geometry_gate():
    for slope in (0,.08,-.12):
        im=np.full((640,640,3),70,np.uint8)
        cv2.line(im,(0,280),(639,round(280+639*slope)),(245,245,245),12)
        cv2.rectangle(im,(490,0),(505,639),(245,245,245),-1)
        paint=np.uint8(im[:,:,0]>200)*255
        result=suppress_markings(im,paint,config())
        np.testing.assert_array_equal(result.binary[:,490:506],paint[:,490:506])
        assert np.count_nonzero(result.binary[:,:470])<.02*np.count_nonzero(paint[:,:470])
        assert not np.any(result.removed & cv2.bitwise_not(paint))


def test_crosswalk_group_suppression_is_preserved_by_geometry_gate():
    im=np.full((640,640,3),70,np.uint8)
    for x in (150,250,350,450):cv2.rectangle(im,(x,150),(x+35,350),(245,245,245),-1)
    cv2.rectangle(im,(570,0),(585,639),(245,245,245),-1)
    paint=np.uint8(im[:,:,0]>200)*255
    result=suppress_markings(im,paint,config())
    assert result.stripe_count==4
    assert not result.binary[150:351,150:486].any()
    np.testing.assert_array_equal(result.binary[:,570:586],paint[:,570:586])
