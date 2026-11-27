"""Sliding-window chains linking observed paint fragments across dashed gaps.

Nodes are actual connected paint fragments inside each vertical window. Links use
the local slope at BOTH ends, so a gap does not force constant-x extrapolation.
The model is pure CPU / OpenCV / NumPy, independent of the application or ROS.
"""
from dataclasses import dataclass
from copy import deepcopy
import cv2
import numpy as np
from .lane_detection import LaneFit,LaneResult
from .supported_lane_detection import fit_supported_rows


@dataclass
class Node:
    level:int
    label:int
    x:float
    y:float
    slope:float
    rows:np.ndarray
    centers:np.ndarray
    pixels:np.ndarray


def observed_error(a,b,min_overlap=40):
    lo=max(a.y_min,b.y_min);hi=min(a.y_max,b.y_max)
    if hi-lo<min_overlap:return None
    y=np.linspace(lo,hi,24)
    return float(np.median(abs(a.x(y)-b.x(y))))


class WindowChainDetector:
    def __init__(self,config,validation):
        self.cfg=deepcopy(config);self.validation=deepcopy(validation)
        self.previous={};self.frame=0;self.width=config['nominal_lane_width_px']
        self.nodes=[];self.candidates=[];self.rejections=[]

    def _nodes(self,binary):
        h,w=binary.shape
        n,labels,stats,_=cv2.connectedComponentsWithStats(binary)
        nodes=[]
        for label in range(1,n):
            x0,y0,cw,ch,area=stats[label]
            if ch<self.cfg['min_node_rows'] or area<self.cfg['row_min_width_px']*self.cfg['min_node_rows']:continue
            yy,xx=np.nonzero(labels[y0:y0+ch,x0:x0+cw]==label);yy+=y0;xx+=x0
            counts=np.bincount(yy,minlength=h);sums=np.bincount(yy,weights=xx,minlength=h)
            minima=np.full(h,100000);maxima=np.full(h,-1)
            starts=np.r_[0,np.flatnonzero(np.diff(yy))+1]
            minima[yy[starts]]=np.minimum.reduceat(xx,starts)
            maxima[yy[starts]]=np.maximum.reduceat(xx,starts)
            good=(counts>=self.cfg['row_min_width_px']) & (maxima-minima+1<=self.cfg['row_max_width_px'])
            rows=np.flatnonzero(good);centers=sums[rows]/counts[rows]
            for level in range(self.cfg['windows']):
                lo=h-(level+1)*h//self.cfg['windows'];hi=h-level*h//self.cfg['windows']
                ix=(rows>=lo)&(rows<hi);ys=rows[ix];xs=centers[ix]
                if len(ys)<self.cfg['min_node_rows']:continue
                ybar=float(ys.mean());xbar=float(xs.mean());dy=ys-ybar
                slope=float(np.dot(dy,xs-xbar)/max(float(np.dot(dy,dy)),1))
                rms=float(np.sqrt(np.mean((xs-(xbar+slope*dy))**2)))
                if rms>5 or abs(slope)>3:continue
                # ys is exactly the supported rows in this disjoint window.
                ids=good[yy]&(yy>=lo)&(yy<hi)
                nodes.append(Node(level,label,xbar,ybar,slope,ys,xs,np.column_stack((xx[ids],yy[ids]))))
        return sorted(nodes,key=lambda node:(node.level,node.x))

    def _link(self,a,b):
        gap=b.level-a.level;dy=b.y-a.y
        if not 1<=gap<=self.cfg['max_link_gap_windows']:return None
        error=abs((b.x-a.x)-.5*(a.slope+b.slope)*dy)
        if error>self.cfg['link_margin_px']+.06*abs(dy):return None
        if abs(a.slope-b.slope)/max(abs(dy),1)>self.cfg['max_slope_change_per_px']:return None
        return .6*error+8*abs(np.arctan(a.slope)-np.arctan(b.slope))+3*(gap-1)

    def _fit_paths(self,binary):
        h=binary.shape[0];nodes=self._nodes(binary);self.nodes=nodes
        beams=[];all_paths=[]
        for j,b in enumerate(nodes):
            options=[(float(len(b.rows)),(j,))]
            for i,a in enumerate(nodes[:j]):
                penalty=self._link(a,b)
                if penalty is None:continue
                for score,path in beams[i]:
                    options.append((score+len(b.rows)-penalty+(3 if a.label==b.label else 0),path+(j,)))
            options.sort(key=lambda item:item[0],reverse=True)
            beams.append(options[:self.cfg['beam_per_node']])
            all_paths.extend(options[:self.cfg['beam_per_node']])
        all_paths.sort(key=lambda item:item[0],reverse=True)
        candidates=[]
        for score,path in all_paths:
            if len(path)<self.cfg['min_occupied_windows']:continue
            chain=[nodes[i] for i in path]
            if sum(len(n.rows) for n in chain)<self.validation['min_supported_rows']:continue
            # Avoid fitting the many prefixes of an already represented chain.
            if any(len(set(path)&set(old.chain_indices))/len(path)>.8 for old in candidates):continue
            pixels=np.concatenate([node.pixels for node in chain])
            fit,reason,detail=fit_supported_rows(pixels[:,0],pixels[:,1],self.cfg,self.validation,h)
            if fit is None:
                self.rejections.append(reason);continue
            fit.chain_indices=path;fit.chain=chain;fit.raw_coefficients=fit.coefficients.copy()
            rows=np.unique(fit.pixels[:,1]);gap=int(np.max(np.diff(rows)-1)) if len(rows)>1 else 0
            fit.dashed=gap>=20 and detail['support_fraction']<.88
            if any((error:=observed_error(fit,old,40)) is not None and error<18 for old in candidates):continue
            candidates.append(fit)
            if len(candidates)>=10:break
        return candidates

    def _history_error(self,fit,side):
        entry=self.previous.get(side)
        if entry is None or self.frame-entry[1]>self.cfg['max_missed_frames']:return None
        return observed_error(fit,entry[0],40)

    def _smooth(self,fit,side):
        raw=fit.coefficients.copy();entry=self.previous.get(side)
        if entry is not None and self.frame-entry[1]<=self.cfg['max_missed_frames']:
            err=observed_error(fit,entry[0],60)
            if err is not None and err<20:
                y=np.unique(fit.pixels[:,1]);delta=entry[0].coefficients-raw
                shift=float(np.max(abs(np.polyval(delta,y))))
                weight=min(.25,self.cfg['bounded_smoothing_px']/max(shift,1e-6))
                proposed=raw+weight*delta
                py=fit.pixels[:,1].astype(int)
                centers=np.bincount(py,weights=fit.pixels[:,0])[y]/np.bincount(py)[y]
                rms=float(np.sqrt(np.mean((centers-np.polyval(proposed,y))**2)))
                if rms<=max(3,fit.residual_px+1):fit.coefficients=proposed;fit.residual_px=rms
        fit.raw_coefficients=raw

    def detect(self,binary,paint=None):
        self.frame+=1;self.rejections=[];h,w=binary.shape
        candidates=self._fit_paths(binary);self.candidates=candidates
        if paint is not None:
            # Classify gaps on original paint, not holes introduced by preprocessing.
            for fit in candidates:
                ys=np.arange(int(fit.y_min),int(fit.y_max)+1)
                xs=np.rint(fit.x(ys)).astype(int)
                hits=np.zeros(len(ys),dtype=bool)
                for offset in range(-4,5):
                    xx=xs+offset;inside=(xx>=0)&(xx<w)
                    hits[inside]|=paint[ys[inside],xx[inside]]!=0
                absent=np.flatnonzero(~hits)
                runs=np.split(absent,np.flatnonzero(np.diff(absent)>1)+1) if len(absent) else []
                max_gap=max((len(run) for run in runs),default=0)
                fit.dashed=max_gap>=20 and float(hits.mean())<.88
                fit.paint_support_fraction=float(hits.mean())
        best=None
        for a in candidates:
            for b in candidates:
                if a is b:continue
                lo=max(a.y_min,b.y_min);hi=min(a.y_max,b.y_max)
                if hi-lo<self.cfg['min_overlap_px']:continue
                widths=b.x(np.linspace(lo,hi,20))-a.x(np.linspace(lo,hi,20))
                if min(widths)<self.cfg['width_range_px'][0] or max(widths)>self.cfg['width_range_px'][1] or np.ptp(widths)>self.cfg['max_width_variation_px']:continue
                if self.cfg['target_lane']=='right_of_dashed' and b.dashed and not a.dashed:continue
                history=[self._history_error(a,'left'),self._history_error(b,'right')]
                if any(e is not None and e>self.cfg['max_temporal_jump_px'] for e in history):continue
                score=(a.support['rows']+b.support['rows'])/h
                score-=abs(float(np.median(widths))-self.width)/self.width
                score-=.03*(a.residual_px+b.residual_px)
                score-=sum(e or 0 for e in history)/160
                if a.dashed:score+=.2
                if best is None or score>best[0]:best=(score,a,b,float(np.median(widths)))
        left=right=None;status='FAILED';reason='no supported window chain'
        if best:
            _,left,right,width=best;self.width=.97*self.width+.03*width
            status='DETECTED';reason='two observed window chains; overlap geometry passed'
        elif candidates:
            choices=[]
            for fit in candidates:
                errors={s:self._history_error(fit,s) for s in ('left','right')}
                known=[(error,s) for s,error in errors.items() if error is not None and error<20]
                locked=min(known)[1] if known else None
                for side in ('left','right'):
                    error=errors[side]
                    if locked is not None and side!=locked:continue
                    # The recorded target is the lane right of the dashed divider.
                    if fit.dashed and self.cfg['target_lane']=='right_of_dashed' and side=='right' and locked!='right':continue
                    if error is not None and error>self.cfg['max_temporal_jump_px']:continue
                    bottom=float(fit.x(min(fit.y_max,h-1)))
                    anchor=w/2+(-.5 if side=='left' else .5)*self.width
                    distance=error if error is not None else abs(bottom-anchor)
                    score=fit.support['rows']/h-.003*distance-.025*fit.residual_px
                    if fit.dashed and side=='left':score+=.35
                    choices.append((score,side,fit))
            if choices:
                _,side,fit=max(choices,key=lambda item:item[0])
                co=fit.coefficients.copy();co[-1]+=self.width if side=='left' else -self.width
                inferred=LaneFit(co,np.empty((0,2)),fit.residual_px,fit.coverage,fit.y_min,fit.y_max,True)
                left,right=(fit,inferred) if side=='left' else (inferred,fit)
                status='INFERRED_RIGHT' if side=='left' else 'INFERRED_LEFT'
                reason='one observed window chain; partner uses pixel-width assumption'
        for side,fit in (('left',left),('right',right)):
            if fit is not None and not fit.inferred:
                self._smooth(fit,side);self.previous[side]=(fit,self.frame)
        # Reconstruct the inferred partner AFTER bounded smoothing, not independently.
        if left is not None and left.inferred:
            left.coefficients=right.coefficients.copy();left.coefficients[-1]-=self.width
        if right is not None and right.inferred:
            right.coefficients=left.coefficients.copy();right.coefficients[-1]+=self.width
        debug=cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR);windows=[]
        for fit,color in ((left,(255,170,30)),(right,(20,225,255))):
            if fit is None or fit.inferred:continue
            px=fit.pixels.astype(int);debug[px[:,1],px[:,0]]=color
            for node in fit.chain:
                lo=h-(node.level+1)*h//self.cfg['windows'];hi=h-node.level*h//self.cfg['windows']
                margin=self.cfg['margin_px'];bounds=(int(node.x-margin),lo,int(node.x+margin),hi)
                windows.append(bounds);cv2.rectangle(debug,bounds[:2],bounds[2:],(60,140,60),1)
        confidence=0 if left is None else min(left.coverage,right.coverage)*np.exp(-max(left.residual_px,right.residual_px)/8)
        if status!='DETECTED':confidence*=.5
        histogram=np.count_nonzero(binary[int(h*.35):],axis=0).astype(float)
        return LaneResult(left,right,status,reason,float(confidence),debug,histogram,windows)
