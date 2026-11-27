from pathlib import Path
import cv2
import numpy as np
import yaml
from dolbat_core.final_lane_detection import FinalLaneDetector
from dolbat_core.window_chain_detection import WindowChainDetector

ROOT=Path(__file__).resolve().parents[1]


def config():return yaml.safe_load((ROOT/'dolbat_core/resources/config/geometry.yaml').read_text(encoding='utf-8'))


def detector():
    g=config();return FinalLaneDetector(g['lane_detection'],g['lane_validation'])


def test_disjoint_new_fragment_does_not_replace_current_tracked_boundary():
    mask=np.zeros((640,640),np.uint8);cv2.rectangle(mask,(494,210),(506,559),255,-1)
    det=detector();first=det.detect(mask);assert first.status=='INFERRED_LEFT'
    for y in (0,130):cv2.rectangle(mask,(134,y),(146,y+70),255,-1)
    result=det.detect(mask)
    assert result.status=='INFERRED_LEFT' and not result.right.inferred
    assert any('disjoint' in reason for reason in result.context_rejections)
    assert abs(result.right.x(350)-500)<1


def test_crosswalk_bar_cannot_become_a_second_boundary():
    mask=np.zeros((640,640),np.uint8)
    cv2.rectangle(mask,(494,0),(506,639),255,-1)
    cv2.rectangle(mask,(134,120),(146,319),255,-1)
    crosswalk=np.zeros_like(mask)
    for x in (220,320,420):cv2.rectangle(crosswalk,(x,120),(x+35,319),255,-1)
    result=detector().detect(mask,crosswalk=crosswalk)
    assert result.status=='INFERRED_LEFT' and result.partial_left is None
    assert any('crosswalk' in reason for reason in result.context_rejections)


def test_real_dashes_outside_crosswalk_band_are_retained():
    mask=np.zeros((640,640),np.uint8);cv2.rectangle(mask,(494,0),(506,639),255,-1)
    for y in (0,170,340,510):cv2.rectangle(mask,(134,y),(146,min(639,y+80)),255,-1)
    crosswalk=np.zeros_like(mask);crosswalk[150:300,220:400]=255
    result=detector().detect(mask,crosswalk=crosswalk)
    assert result.status=='DETECTED'


def test_short_right_fragment_is_partial_not_a_full_observed_pair():
    mask=np.zeros((640,640),np.uint8)
    for y in (0,200,400):cv2.rectangle(mask,(134,y),(146,y+80),255,-1)
    cv2.rectangle(mask,(494,440),(506,550),255,-1)
    result=detector().detect(mask)
    assert result.status=='INFERRED_RIGHT' and result.right.inferred
    part=result.partial_right
    assert part is not None and part.observation_kind=='PARTIAL'
    assert not part.inferred and len(part.pixels)>0
    assert np.all(mask[part.pixels[:,1].astype(int),part.pixels[:,0].astype(int)]!=0)
    assert part.y_min>=440 and part.y_max<=550


def test_blank_frame_has_no_stale_full_or_partial_observations():
    det=detector();mask=np.zeros((640,640),np.uint8)
    cv2.rectangle(mask,(134,0),(146,639),255,-1);det.detect(mask)
    result=det.detect(np.zeros_like(mask))
    assert not result.usable and result.left is None and result.right is None
    assert result.partial_left is None and result.partial_right is None
