"""The stateful stages reject gaps and changed recording metadata."""
import pytest
rclpy=pytest.importorskip('rclpy')
pytest.importorskip('dolbat_interfaces.msg')
from dolbat_interfaces.msg import VideoFrame
from dolbat_pipeline.common import OrderedStage


@pytest.fixture
def stage():
    rclpy.init();node=OrderedStage('ordered_stage_test')
    try:yield node
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


def message(index,fps=24.,eos=False):
    msg=VideoFrame();msg.frame_index=index;msg.source_fps=fps;msg.end_of_stream=eos
    return msg


def test_ordered_frames_and_eos(stage):
    assert stage.accept(message(0))
    assert stage.accept(message(1))
    assert not stage.accept(message(2,eos=True))
    with pytest.raises(RuntimeError,match='after end-of-stream'):stage.accept(message(2))


def test_gap_is_not_silently_used_as_the_next_tracking_frame(stage):
    stage.accept(message(0))
    with pytest.raises(RuntimeError,match='expected frame 1'):stage.accept(message(2))


def test_source_fps_cannot_change_mid_recording(stage):
    stage.accept(message(0))
    with pytest.raises(ValueError,match='FPS changed'):stage.accept(message(1,30.))
