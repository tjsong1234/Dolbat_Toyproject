import time
import numpy as np
import pytest
from dolbat_core.video_output import AsyncVideoSink


class Writer:
    def __init__(self,fail=False,opened=True):self.values=[];self.fail=fail;self.opened=opened;self.released=False
    def isOpened(self):return self.opened
    def write(self,image):
        if self.fail:raise OSError('disk failure')
        time.sleep(.001);self.values.append(int(image[0,0,0]))
    def release(self):self.released=True


def test_backpressure_preserves_pair_order_and_every_frame(tmp_path):
    writers=[Writer(),Writer()];created=[]
    def factory(*args):
        result=writers[len(created)];created.append(result);return result
    sink=AsyncVideoSink([(tmp_path/'a.mp4',(3,2)),(tmp_path/'b.mp4',(3,2))],24,max_pending=1,writer_factory=factory)
    for i in range(12):sink.write(np.full((2,3,3),i,np.uint8),np.full((2,3,3),i+20,np.uint8))
    sink.close()
    assert writers[0].values==list(range(12)) and writers[1].values==list(range(20,32))
    assert all(w.released for w in writers) and sink.stats()['written_pairs']==12
    assert sink.stats()['frames_dropped']==0


def test_writer_failure_is_raised_and_thread_terminates(tmp_path):
    sink=AsyncVideoSink([(tmp_path/'a.mp4',(3,2))],24,writer_factory=lambda *a:Writer(fail=True))
    sink.write(np.zeros((2,3,3),np.uint8))
    with pytest.raises(RuntimeError,match='failed'):sink.close()
    assert not sink.thread.is_alive()


def test_open_failure_is_raised(tmp_path):
    with pytest.raises(RuntimeError,match='failed'):
        AsyncVideoSink([(tmp_path/'a.mp4',(3,2))],24,writer_factory=lambda *a:Writer(opened=False))


def test_invalid_frame_is_rejected_without_queueing(tmp_path):
    sink=AsyncVideoSink([(tmp_path/'a.mp4',(3,2))],24,writer_factory=lambda *a:Writer())
    with pytest.raises(ValueError,match='shape'):sink.write(np.zeros((3,3,3),np.uint8))
    sink.close();assert sink.written==0


def test_snapshot_is_saved_before_close_returns(tmp_path):
    import cv2
    image=np.full((2,3,3),87,np.uint8);path=tmp_path/'snapshot.png'
    sink=AsyncVideoSink([(tmp_path/'a.mp4',(3,2))],24,writer_factory=lambda *a:Writer())
    sink.write(image,pngs=[(path,image)])
    sink.close()
    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR),image)
    assert sink.stats()['snapshot_pngs_written']==1
