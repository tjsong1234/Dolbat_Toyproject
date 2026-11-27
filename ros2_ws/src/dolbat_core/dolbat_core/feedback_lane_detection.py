"""Wider reacquisition only after a normal search is rejected.

Both searches use the SAME pre-frame temporal state and observation gates. A failed
retry must never count as a second missed video frame, or replace a valid primary fit.
"""
from copy import deepcopy
from .supported_lane_detection import SupportedLaneDetector


class FeedbackLaneDetector(SupportedLaneDetector):
    def __init__(self,config,validation,retry_margin_px=115,trace=False):
        super().__init__(deepcopy(config),validation,trace)
        self.retry_margin=retry_margin_px
        self.retries=0
        self.retry_successes=0
        if retry_margin_px<=self.cfg['margin_px']:
            raise ValueError('Retry margin must exceed normal margin')

    def _state(self):
        return self.previous,self.width,self.missed,self.rejections,self.accepted_fits,self.trace

    def _restore(self,state):
        self.previous,self.width,self.missed,self.rejections,self.accepted_fits,self.trace=state

    def detect(self,binary):
        margin=self.cfg['margin_px'];before=self._state()
        primary=super().detect(binary)
        primary.search_margin_px=margin
        primary.retry_attempted=False;primary.retry_accepted=False
        if primary.usable:
            return primary
        after=self._state();self.retries+=1
        self._restore(before);self.cfg['margin_px']=self.retry_margin
        try:
            retry=super().detect(binary)
        finally:
            self.cfg['margin_px']=margin
        if retry.usable:
            self.retry_successes+=1
            retry.search_margin_px=self.retry_margin
            retry.retry_attempted=True;retry.retry_accepted=True
            return retry
        self._restore(after)
        primary.retry_attempted=True
        return primary
