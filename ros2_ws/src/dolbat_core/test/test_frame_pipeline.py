from types import SimpleNamespace
import numpy as np
import pytest
from dolbat_core.frame_pipeline import OrderedPreprocessor


class Capture:
    def __init__(self,n=8):self.n=n;self.index=0;self.released=False
    def read(self):
        if self.index==self.n:return False,None
        value=self.index;self.index+=1;return True,np.full((2,3,3),value,np.uint8)
    def release(self):self.released=True


class Preprocessor:
    def process(self,frame):
        self.paint=frame[:,:,0].copy()
        self.suppression=SimpleNamespace(crosswalk=self.paint+20)
        self.matrix=np.eye(3)*(1+int(frame[0,0,0]));self.inverse=self.matrix.copy();self.input_size=(3,2)
        return SimpleNamespace(binary=self.paint+10)


def test_order_and_packet_masks_do_not_follow_mutable_latest_attributes():
    cap=Capture();pre=Preprocessor();worker=OrderedPreprocessor(cap,pre)
    packets=[]
    try:
        while True:
            packet=worker.next()
            if packet is None:break
            packets.append(packet)
    finally:worker.close()
    assert len(packets)==8 and cap.released
    for i,packet in enumerate(packets):
        assert packet.index==i
        assert np.all(packet.paint==i) and np.all(packet.crosswalk==i+20)
        assert np.all(packet.processed.binary==i+10)
        assert packet.matrix[0,0]==i+1
    assert worker.stats()['produced']==worker.stats()['consumed']==8


def test_worker_failure_is_not_reported_as_end_of_video():
    class Broken(Preprocessor):
        def process(self,frame):raise OSError('bad preprocessing')
    worker=OrderedPreprocessor(Capture(),Broken())
    with pytest.raises(RuntimeError,match='worker failed'):worker.next()
    with pytest.raises(RuntimeError,match='worker failed'):worker.close()
    assert not worker.thread.is_alive()


def test_early_close_unblocks_a_full_pending_queue():
    cap=Capture(500);worker=OrderedPreprocessor(cap,Preprocessor())
    packet=worker.next();assert packet.index==0
    worker.close()
    assert cap.released and not worker.thread.is_alive()
