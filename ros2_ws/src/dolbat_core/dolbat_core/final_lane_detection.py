"""Context checks and explicit partial observations for the week-3 detector.

The main lane acceptance gates remain those of WindowChainDetector. Short current
fragments stay separate from full boundaries. A planner may use their actual
rows as explicitly weak width evidence without promoting the detection state.
"""
from collections import defaultdict
from copy import deepcopy
import numpy as np
from .window_chain_detection import WindowChainDetector,observed_error
from .supported_lane_detection import fit_supported_rows


class FinalLaneDetector(WindowChainDetector):
    def __init__(self,config,validation,context=None):
        super().__init__(config,validation)
        self.context={
            'history_match_px':20.,'crosswalk_row_min_pixels':25,
            'crosswalk_max_support_fraction':.85,
            'partial_min_rows':30,'partial_min_span_px':35.,
            'partial_min_overlap_px':20.,'partial_max_residual_px':3.,
            'partial_max_slope_difference_rad':.45,
            **(context or {})}
        self._crosswalk_rows=None;self.context_rejections=[]

    def _in_crosswalk_band(self,fit):
        if self._crosswalk_rows is None or not np.any(self._crosswalk_rows):return False
        rows=np.unique(fit.pixels[:,1].astype(int))
        if not len(rows):return False
        fraction=float(np.mean(self._crosswalk_rows[rows]))
        fit.crosswalk_row_fraction=fraction
        return fraction>=self.context['crosswalk_max_support_fraction']

    def _fit_paths(self,binary):
        candidates=super()._fit_paths(binary)
        kept=[]
        for fit in candidates:
            if self._in_crosswalk_band(fit):
                self.context_rejections.append('candidate supported almost entirely inside crosswalk band')
            else:kept.append(fit)
        # A newly appearing far-only fragment must not replace a currently seen
        # boundary already associated with fresh history. No stale fit is returned:
        # anchored entries here are actual candidates measured in THIS frame.
        anchored=[]
        for fit in kept:
            errors=[self._history_error(fit,side) for side in ('left','right')]
            if any(e is not None and e<self.context['history_match_px'] for e in errors):anchored.append(fit)
        if anchored:
            accepted=[]
            for fit in kept:
                if any(fit is old for old in anchored):
                    accepted.append(fit);continue
                overlaps=[min(fit.y_max,a.y_max)-max(fit.y_min,a.y_min) for a in anchored]
                if max(overlaps,default=0)<self.cfg['min_overlap_px']:
                    self.context_rejections.append('new disjoint fragment cannot replace a currently associated observation')
                else:accepted.append(fit)
            kept=accepted
        return kept

    def _partial_observations(self,result,binary):
        parts={'left':None,'right':None}
        observed={side:fit for side,fit in (('left',result.left),('right',result.right))
                  if fit is not None and not fit.inferred}
        if not observed:return parts
        used={node.label for fit in observed.values() for node in getattr(fit,'chain',[])}
        by_label=defaultdict(list)
        for node in self.nodes:
            if node.label not in used:by_label[node.label].append(node)
        height=binary.shape[0]
        cfg=deepcopy(self.cfg);val=deepcopy(self.validation)
        cfg['min_y_coverage']=self.context['partial_min_span_px']/height
        cfg['min_occupied_windows']=2;cfg['max_residual_px']=self.context['partial_max_residual_px']
        val.update(min_supported_rows=self.context['partial_min_rows'],min_support_fraction=.75,
                   max_unsupported_gap_px=20,min_rows_per_half=8)
        choices={side:[] for side in parts}
        for nodes in by_label.values():
            if len(nodes)<2:continue
            pixels=np.concatenate([n.pixels for n in nodes])
            fit,_,_=fit_supported_rows(pixels[:,0],pixels[:,1],cfg,val,height)
            if fit is None or self._in_crosswalk_band(fit):continue
            fit.observation_kind='PARTIAL';fit.chain=nodes
            fit.partial_reason='current fragment has local support; full-boundary gates not satisfied or context not sufficient'
            for side in ('left','right'):
                if side in observed:continue
                other=observed.get('right' if side=='left' else 'left')
                if other is None:continue
                lo=max(fit.y_min,other.y_min);hi=min(fit.y_max,other.y_max)
                if hi-lo<self.context['partial_min_overlap_px']:continue
                yy=np.linspace(lo,hi,24)
                widths=other.x(yy)-fit.x(yy) if side=='left' else fit.x(yy)-other.x(yy)
                if widths.min()<self.cfg['width_range_px'][0] or widths.max()>self.cfg['width_range_px'][1]:continue
                first=np.polyval(np.polyder(fit.coefficients),yy)
                second=np.polyval(np.polyder(other.coefficients),yy)
                if np.median(abs(np.arctan(first)-np.arctan(second)))>self.context['partial_max_slope_difference_rad']:continue
                score=fit.support['rows']-3*fit.residual_px-.1*abs(float(np.median(widths))-self.width)
                choices[side].append((score,fit))
        for side in parts:
            if choices[side]:parts[side]=max(choices[side],key=lambda item:item[0])[1]
        return parts

    def detect(self,binary,paint=None,crosswalk=None):
        self.context_rejections=[]
        self._crosswalk_rows=(np.count_nonzero(crosswalk,axis=1)>=self.context['crosswalk_row_min_pixels']) if crosswalk is not None else None
        result=super().detect(binary,paint)
        partial=self._partial_observations(result,binary)
        result.partial_left=partial['left'];result.partial_right=partial['right']
        result.context_rejections=list(self.context_rejections)
        result.partial_summary={side:{'rows':fit.support['rows'],'y_min':fit.y_min,'y_max':fit.y_max,
            'pixel_count':len(fit.pixels),'kind':'PARTIAL'} for side,fit in partial.items() if fit is not None}
        return result
