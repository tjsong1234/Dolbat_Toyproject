from pathlib import Path
from itertools import permutations
import cv2
import numpy as np
import pytest
import yaml
from dolbat_core.roi_editor import order_quad,validate_quad,make_bev,save_roi_config


def test_click_order_does_not_block_a_valid_roi():
    points = [[110,248],[416,240],[585,479],[0,479]]
    for clicked in permutations(points):
        ordered = order_quad(clicked)
        np.testing.assert_allclose(ordered,points)
        assert validate_quad(ordered,640,480)[0]


def test_quad_validity():
    assert validate_quad([[20,20],[80,20],[99,99],[0,99]],100,100)[0]
    assert not validate_quad([[0,0],[99,99],[99,0],[0,99]],100,100)[0]
    assert not validate_quad([[0,0],[0,99],[99,99],[99,0]],100,100)[0]
    assert not validate_quad([[0,0],[99,0],[99,99]],100,100)[0]
    assert not validate_quad([[-1,0],[99,0],[99,99],[0,99]],100,100)[0]


def test_transform_corners_and_content():
    frame = np.zeros((100,100,3),np.uint8)
    cv2.rectangle(frame,(20,20),(80,80),(255,255,255),-1)
    bev,_ = make_bev(frame,[[20,20],[80,20],[80,80],[20,80]],(60,80))
    assert bev.shape == (80,60,3)
    assert np.all(bev == 255)
    assert make_bev(frame,[],(60,80))[0] is None


def test_save_and_backup(tmp_path):
    config = tmp_path/'settings.yaml'
    cfg = {'preprocessing':{'source_points':[], 'destination_points':[], 'bev_size':[640,480]},
           'calibration':{'status':'assumed_not_measured'}}
    config.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    original = config.read_bytes()
    points = [[20,20],[80,20],[99,99],[0,99]]
    backup = save_roi_config(config,points,(100,100),Path('original.mp4'),120)
    saved = yaml.safe_load(config.read_text(encoding='utf-8'))
    assert backup.read_bytes() == original
    assert saved['calibration'] == cfg['calibration']
    np.testing.assert_allclose(np.array(saved['preprocessing']['source_points'])*99,points)
    assert saved['roi_selection']['frame_index_zero_based'] == 120
    assert saved['preprocessing']['destination_points'] == [[0,0],[1,0],[1,1],[0,1]]
    with pytest.raises(ValueError):
        save_roi_config(config,[],(100,100),'original.mp4',0)
