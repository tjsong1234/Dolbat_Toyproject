"""Connected-road filtering must reject background paint without cutting lanes."""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

from dolbat_core.geometry_preprocessing import GeometryPreprocessor
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.road_surface_preprocessing import RoadSurfacePreprocessor


ROOT=Path(__file__).resolve().parents[1]


def configuration():
    base=yaml.safe_load((ROOT/'dolbat_core/resources/config/settings.yaml').read_text(encoding='utf-8'))['preprocessing']
    geometry=yaml.safe_load((ROOT/'dolbat_core/resources/config/geometry_final.yaml').read_text(encoding='utf-8'))
    marking=yaml.safe_load((ROOT/'dolbat_core/resources/config/perception_improved.yaml').read_text(encoding='utf-8'))['marking_suppression']
    return base,geometry,marking


def identity_preprocessor(cls=RoadSurfacePreprocessor):
    """Use the real color/thickness settings, with identity image coordinates."""
    base,geometry,marking=configuration()
    base=deepcopy(base);geometry=deepcopy(geometry)
    base['source_points']=[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]
    geometry['preprocessing'].update(
        processing_width=640,bev_size=[640,640],
        roi_polygon=[[0.,0.],[1.,0.],[1.,1.],[0.,1.]],
        destination_points=[[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    return cls(base,geometry,marking)


def road_fixture():
    # Value 160 lies between road_max_value=155 and white_min_value=165.
    # The neutral floor therefore belongs to neither the road nor the paint.
    frame=np.full((640,640,3),160,np.uint8)
    cv2.rectangle(frame,(70,0),(430,639),(70,70,70),-1)
    cv2.rectangle(frame,(80,0),(94,639),(245,245,245),-1)
    cv2.rectangle(frame,(410,0),(424,639),(245,245,245),-1)
    return frame


def test_disconnected_shadow_cannot_make_background_paint_a_lane():
    frame=road_fixture()
    cv2.rectangle(frame,(500,430),(550,530),(70,70,70),-1)
    cv2.rectangle(frame,(535,435),(546,525),(245,245,245),-1)
    previous=identity_preprocessor(GeometryPreprocessor);previous.process(frame)
    assert previous.candidate[435:526,535:547].all(), 'The fixture must pass simple local road adjacency'

    current=identity_preprocessor();result=current.process(frame)
    assert not current.candidate[435:526,535:547].any()
    assert not result.binary[435:526,535:547].any()
    # Preserve the entire width and length of each real boundary, not just edges.
    np.testing.assert_array_equal(result.binary[:,80:95],previous.candidate[:,80:95])
    np.testing.assert_array_equal(result.binary[:,410:425],previous.candidate[:,410:425])
    assert result.binary[:,85:90].all() and result.binary[:,415:420].all()


def test_stop_bar_does_not_disconnect_the_road_or_remove_upstream_lane():
    frame=road_fixture()
    cv2.rectangle(frame,(70,350),(430,370),(245,245,245),-1)
    current=identity_preprocessor();result=current.process(frame)
    # The stop bar spans the complete road. Pure dark-pixel connectivity would
    # split it, keeping the lower component and losing the lanes above the bar.
    assert current.road_surface[100,250] and current.road_surface[500,250]
    assert current.road_surface[360,250], 'Observed paint must bridge surface connectivity'
    assert result.binary[100:300,80:95].all()
    assert result.binary[450:600,80:95].all()
    assert result.binary[350:371,80:95].all(), 'The crossing longitudinal paint must survive'
    assert not result.binary[350:371,150:350].any(), 'The actual stop bar is still suppressed'


def test_no_dark_road_does_not_promote_an_isolated_white_component():
    frame=np.full((640,640,3),160,np.uint8)
    cv2.rectangle(frame,(300,20),(315,620),(245,245,245),-1)
    current=identity_preprocessor();result=current.process(frame)
    assert not current.candidate.any() and not result.binary.any()
    blank=current.process(np.zeros_like(frame))
    assert not blank.binary.any() and not current.road_surface.any()


def test_bev_side_padding_is_only_a_translation_and_preserves_cm_coordinates():
    base,final,marking=configuration()
    earlier=yaml.safe_load((ROOT/'dolbat_core/resources/config/geometry.yaml').read_text(encoding='utf-8'))
    assert final['preprocessing']['roi_polygon']==earlier['preprocessing']['roi_polygon']
    frame=np.full((480,640,3),70,np.uint8)
    cv2.line(frame,(100,479),(240,220),(245,245,245),10)
    cv2.line(frame,(550,479),(380,220),(245,245,245),10)
    old=RoadSurfacePreprocessor(base,earlier,marking);p_old=old.process(frame)
    padded=RoadSurfacePreprocessor(base,final,marking);p_new=padded.process(frame)
    np.testing.assert_array_equal(p_new.roi,p_old.roi)
    np.testing.assert_array_equal(padded.anchor_points,old.anchor_points)
    points=np.float64([[80,240],[320,240],[550,260],[100,430],[320,470],[600,400]])
    uv_old=cv2.perspectiveTransform(points.reshape(-1,1,2),old.matrix).reshape(-1,2)
    uv_new=cv2.perspectiveTransform(points.reshape(-1,1,2),padded.matrix).reshape(-1,2)
    # OpenCV builds each homography from float32 anchors; allow sub-millipixel
    # rounding while still detecting any accidental scale or origin change.
    np.testing.assert_allclose(uv_new-uv_old,np.tile([80.,0.],(len(points),1)),atol=2e-4)
    old_mapping=MetricMapping(.25449101796407186,.47076158562377646,319.5,639)
    new_mapping=MetricMapping(.25449101796407186,.47076158562377646,399.5,639)
    np.testing.assert_allclose(new_mapping.pixels_to_cm(uv_new),old_mapping.pixels_to_cm(uv_old),atol=6e-5)
    np.testing.assert_allclose(new_mapping.cm_to_pixels(new_mapping.pixels_to_cm(uv_new)),uv_new,atol=1e-8)
