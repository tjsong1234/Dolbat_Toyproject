"""Ordered one-worker preprocessing; stateful tracking/planning stay in caller.

Each packet captures its own paint, masks and transforms. Reading attributes of
the mutable preprocessor in the consumer would mix two different frames.
"""
from dataclasses import dataclass
import queue
import threading
import time


@dataclass(frozen=True)
class FramePacket:
    index:int
    frame:object
    processed:object
    paint:object
    crosswalk:object
    matrix:object
    inverse:object
    input_size:tuple
    decode_ms:float
    preprocess_ms:float


class OrderedPreprocessor:
    def __init__(self,capture,preprocessor,max_pending=1):
        if max_pending<1:raise ValueError('Positive pending-frame limit required')
        self.capture=capture;self.pre=preprocessor;self.pending=queue.Queue(maxsize=max_pending)
        self.stop=threading.Event();self.error=None;self.produced=0;self.consumed=0
        self.thread=threading.Thread(target=self._work,name='frame-preprocessing',daemon=True)
        self.thread.start()

    def _put(self,item):
        while not self.stop.is_set():
            try:self.pending.put(item,timeout=.05);return True
            except queue.Full:continue
        return False

    def _work(self):
        try:
            while not self.stop.is_set():
                t0=time.perf_counter();ok,frame=self.capture.read();t1=time.perf_counter()
                if not ok:break
                processed=self.pre.process(frame);t2=time.perf_counter()
                packet=FramePacket(self.produced,frame,processed,self.pre.paint,self.pre.suppression.crosswalk,
                                   self.pre.matrix,self.pre.inverse,self.pre.input_size,(t1-t0)*1000,(t2-t1)*1000)
                if not self._put(packet):return
                self.produced+=1
            self._put(None)
        except BaseException as exc:self.error=exc
        finally:self.capture.release()

    def next(self):
        while True:
            if self.error is not None:raise RuntimeError('Preprocessing worker failed') from self.error
            try:packet=self.pending.get(timeout=.05)
            except queue.Empty:continue
            if packet is not None:
                if packet.index!=self.consumed:raise RuntimeError('Frame order mismatch')
                self.consumed+=1
            return packet

    def close(self):
        self.stop.set();self.thread.join()
        if self.error is not None:raise RuntimeError('Preprocessing worker failed') from self.error

    def stats(self):
        return {'mode':'ordered_preprocessing_worker','max_pending_packets':self.pending.maxsize,
                'produced':self.produced,'consumed':self.consumed,'frames_dropped':0}
