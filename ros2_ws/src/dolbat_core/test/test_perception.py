from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import yaml
from dolbat_core.lane_detection import LaneDetector
from dolbat_core.perception_review import candidate_mask


def lane_config():
    path=Path(__file__).resolve().parents[1]/'dolbat_core/resources/config/settings.yaml'
    return deepcopy(yaml.safe_load(path.read_text(encoding='utf-8'))['lane_detection'])


def test_broken_left_lane_and_continuous_right_lane():
    mask=np.zeros((480,640),np.uint8)
    for y in (0,140,280,420):
        cv2.rectangle(mask,(114,y),(126,min(479,y+75)),255,-1)
    cv2.rectangle(mask,(479,0),(491,479),255,-1)
    detector=LaneDetector(lane_config())
    result=detector.detect(mask)
    assert result.status=='DETECTED'
    assert not result.left.inferred and not result.right.inferred
    assert abs(result.left.x(300)-120)<3
    assert abs(result.right.x(300)-485)<3
    # Previously accepted curves must not be presented as fresh observations on an empty image.
    lost=detector.detect(np.zeros_like(mask))
    assert not lost.usable and lost.left is None and lost.right is None


def test_one_boundary_is_explicitly_inferred():
    mask=np.zeros((480,640),np.uint8)
    cv2.rectangle(mask,(114,0),(126,479),255,-1)
    result=LaneDetector(lane_config()).detect(mask)
    assert result.status=='INFERRED_RIGHT'
    assert result.right.inferred and not result.left.inferred


def test_refined_candidate_removes_thin_strokes_and_retains_thicker_paint():
    bev=np.full((480,640,3),70,np.uint8)
    cv2.rectangle(bev,(100,0),(102,479),(245,245,245),-1)
    cv2.rectangle(bev,(250,0),(264,479),(245,245,245),-1)
    mask=candidate_mask(bev,np.zeros((480,640),np.uint8),'refined',{})
    assert np.count_nonzero(mask[:,98:105])==0
    assert np.count_nonzero(mask[:,250:265])>3000
    uniform=np.full_like(bev,230)
    assert not candidate_mask(uniform,np.zeros_like(mask),'refined',{}).any()
