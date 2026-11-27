"""Bounded, lossless background output of synchronized video frame pairs.

The caller transfers ownership of each array until close(): do not mutate a
submitted image. At most two pending pairs are retained; backpressure waits,
never drops or repeats frames. Errors in the worker are raised in the caller.
"""
import os
from pathlib import Path
import queue
import threading
import time
import cv2
import numpy as np


class AsyncVideoSink:
    def __init__(self, outputs, fps, max_pending=2, writer_factory=None):
        if max_pending<1:raise ValueError('Positive pending-frame limit required')
        self.outputs=[(Path(path),tuple(size)) for path,size in outputs]
        self.fps=fps;self.factory=writer_factory or self._native_writer
        self.pending=queue.Queue(maxsize=max_pending)
        self.ready=threading.Event();self.error=None;self.closed=False
        self.submitted=0;self.written=0;self.write_seconds=0.;self.wait_seconds=0.;self.pngs_written=0
        self.thread=threading.Thread(target=self._work,name='video-output',daemon=True)
        self.thread.start();self.ready.wait()
        self._raise_error()

    @staticmethod
    def _native_writer(path,fps,size):
        return cv2.VideoWriter(os.path.relpath(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)

    def _raise_error(self):
        if self.error is not None:raise RuntimeError('Background video output failed') from self.error

    def _put(self,value):
        while True:
            self._raise_error()
            try:self.pending.put(value,timeout=.05);return
            except queue.Full:continue

    def write(self,*images,pngs=()):
        if self.closed:raise RuntimeError('Video sink is closed')
        if len(images)!=len(self.outputs):raise ValueError('One frame required for each output')
        for image,(_,size) in zip(images,self.outputs):
            if image.shape!=(size[1],size[0],3) or image.dtype!=np.uint8:
                raise ValueError('Video frame shape/dtype mismatch')
        start=time.perf_counter();self._put((images,tuple(pngs)));self.wait_seconds+=time.perf_counter()-start
        self.submitted+=1

    def _work(self):
        writers=[]
        try:
            for path,size in self.outputs:
                writer=self.factory(path,self.fps,size);writers.append(writer)
                if not writer.isOpened():raise OSError(f'Could not open {path}')
            self.ready.set()
            while True:
                item=self.pending.get()
                if item is None:break
                images,pngs=item
                tick=time.perf_counter()
                for writer,image in zip(writers,images):writer.write(image)
                if pngs:
                    from .sliding_visualization import save_png
                    for path,image in pngs:save_png(path,image);self.pngs_written+=1
                self.write_seconds+=time.perf_counter()-tick;self.written+=1
        except BaseException as exc:self.error=exc
        finally:
            self.ready.set()
            for writer in writers:writer.release()

    def close(self):
        if self.closed:
            self._raise_error();return
        self.closed=True
        try:self._put(None)
        finally:self.thread.join()
        self._raise_error()
        if self.written!=self.submitted:raise RuntimeError('Video output frame count mismatch')

    def stats(self):
        return {'mode':'bounded_background_writer','max_pending_pairs':self.pending.maxsize,
                'submitted_pairs':self.submitted,'written_pairs':self.written,
                'worker_write_seconds':self.write_seconds,'producer_wait_seconds':self.wait_seconds,
                'snapshot_pngs_written':self.pngs_written,
                'frames_dropped':0}
