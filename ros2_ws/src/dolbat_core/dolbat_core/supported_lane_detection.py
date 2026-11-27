"""Row-balanced lane fitting and explicit rejection of weak observation support."""
import cv2
import numpy as np
from .lane_detection import LaneDetector,LaneFit


def coherent_rows(rows,validation):
    """Tiny disconnected remnants must not enlarge the apparent observation span."""
    if not len(rows):
        return np.zeros(0,dtype=bool)
    groups=np.split(np.arange(len(rows)),np.flatnonzero(np.diff(rows)>validation.get('row_run_bridge_gap_px',2)+1)+1)
    keep=np.zeros(len(rows),dtype=bool)
    for group in groups:
        if len(group)>=validation.get('min_row_run_length',8):
            keep[group]=True
    return keep


def fit_supported_rows(x,y,cfg,validation,height):
    if len(y)>1 and np.any(y[1:]<y[:-1]):
        order=np.argsort(y,kind='stable');x,y=x[order],y[order]
    counts=np.bincount(y,minlength=height)
    total=np.bincount(y,weights=x,minlength=height)
    minima=np.full(height,np.inf);maxima=np.full(height,-np.inf)
    if len(y):
        starts=np.r_[0,np.flatnonzero(np.diff(y))+1]
        minima[y[starts]]=np.minimum.reduceat(x,starts)
        maxima[y[starts]]=np.maximum.reduceat(x,starts)
    valid=(counts >= validation.get('min_row_pixels',5)) & (maxima-minima <= validation.get('max_row_spread_px',65))
    rows=np.flatnonzero(valid)
    centers=total[valid]/counts[valid]
    coherent=coherent_rows(rows,validation)
    rows,centers=rows[coherent],centers[coherent]
    details={'rows':int(len(rows)),'support_fraction':0.,'max_gap_px':height,'occupied_windows':0}
    if len(rows)<3:
        return None,'too few supported rows',details
    keep=np.ones(len(rows),dtype=bool)
    for _ in range(3):
        if np.count_nonzero(keep)<3:
            return None,'row-center fit lost support',details
        ids=np.minimum(rows[keep]*cfg['windows']//height,cfg['windows']-1)
        per_window=np.bincount(ids,minlength=cfg['windows'])
        weights=1/np.sqrt(per_window[ids])
        co=np.polyfit(rows[keep]/(height-1),centers[keep],2,w=weights)
        error=centers-np.polyval(co,rows/(height-1))
        mad=np.median(abs(error[keep]-np.median(error[keep])))
        keep=abs(error)<max(5.,min(25.,3*1.4826*mad))
    rows,centers=rows[keep],centers[keep]
    coherent=coherent_rows(rows,validation)
    rows,centers=rows[coherent],centers[coherent]
    if len(rows)<3:
        return None,'row-center fit lost support',details
    span=int(rows[-1]-rows[0]+1)
    gap=int(np.max(np.diff(rows)-1)) if len(rows)>1 else height
    window_counts=np.bincount(np.minimum(rows*cfg['windows']//height,cfg['windows']-1),minlength=cfg['windows'])
    occupied=int(np.count_nonzero(window_counts >= validation.get('min_rows_per_window',8)))
    fraction=len(rows)/span
    middle=(rows[0]+rows[-1])/2
    halves=min(np.count_nonzero(rows<=middle),np.count_nonzero(rows>middle))
    details.update(rows=int(len(rows)),support_fraction=float(fraction),max_gap_px=gap,occupied_windows=occupied)
    if len(rows)<validation.get('min_supported_rows',80):
        return None,'insufficient supported rows',details
    if span/height<cfg['min_y_coverage'] or occupied<cfg['min_occupied_windows']:
        return None,'insufficient longitudinal observation',details
    if fraction<validation.get('min_support_fraction',.35):
        return None,'low supported-row fraction',details
    if gap>validation.get('max_unsupported_gap_px',150):
        return None,'unsupported gap too long',details
    if halves<validation.get('min_rows_per_half',18):
        return None,'one half of fit lacks observation',details
    ids=np.minimum(rows*cfg['windows']//height,cfg['windows']-1)
    weights=1/np.sqrt(np.bincount(ids,minlength=cfg['windows'])[ids])
    co=np.polyfit(rows/(height-1),centers,2,w=weights)/[(height-1)**2,height-1,1]
    rms=float(np.sqrt(np.mean((centers-np.polyval(co,rows))**2)))
    if rms>cfg['max_residual_px']:
        return None,'row-center residual too high',details
    supported=np.zeros(height,dtype=bool);supported[rows]=True
    selected=supported[y] & (abs(x-np.polyval(co,y))<=validation.get('max_row_spread_px',65)/2+3)
    fit=LaneFit(co,np.column_stack((x[selected],y[selected])),rms,float((span-1)/(height-1)),float(rows[0]),float(rows[-1]))
    fit.support=details
    return fit,'supported',details


class SupportedLaneDetector(LaneDetector):
    def __init__(self,config,validation,trace=False):
        super().__init__(config)
        self.validation=validation
        self.rejections=[]
        self.accepted_fits=0
        self.trace_enabled=trace
        self.trace={}

    def _trace_selection(self,left,right):
        if not self.trace_enabled:
            return
        self.trace['selected']={side:next((t for t in self.trace['candidates']
            if t.get('fit') is lane),None) if lane is not None and not lane.inferred else None
            for side,lane in (('left',left),('right',right))}

    def _slide(self,binary,start):
        h,w=binary.shape
        coordinates=getattr(self,'_coordinates',None)
        yy,xx=coordinates if coordinates is not None else binary.nonzero()
        indices=[];windows=[];current=float(start);last_center=None;slope=0.
        steps=[]
        for j in range(self.cfg['windows']):
            low,high=h-(j+1)*h//self.cfg['windows'],h-j*h//self.cfg['windows']
            margin=self.cfg['margin_px']
            windows.append((int(current-margin),low,int(current+margin),high))
            ids=np.flatnonzero((yy>=low)&(yy<high)&(abs(xx-current)<=margin))
            before=current;center=None;eligible=np.array([],dtype=int)
            if len(ids)>=self.cfg['min_pixels_window']:
                # Ignore long horizontal rows when choosing the next window center too.
                counts=np.bincount(yy[ids],minlength=h)
                eligible=ids[(counts[yy[ids]]>=self.validation.get('min_row_pixels',5)) &
                             (counts[yy[ids]]<=self.validation.get('max_row_spread_px',65))]
                if len(eligible)>=self.cfg['min_pixels_window']:
                    center=float(np.median(xx[eligible]))
                    if last_center is not None:
                        slope=np.clip(center-last_center,-margin*.8,margin*.8)
                    last_center=center;current=center+slope
                else:
                    current+=slope
                indices.append(ids)
            else:
                current+=slope
            if self.trace_enabled:
                steps.append({'index':j+1,'bounds':windows[-1],'center_before':before,
                              'median':center,'center_next':current,'slope':float(slope),
                              'pixels':np.column_stack((xx[ids],yy[ids])),
                              'eligible_count':int(len(eligible)),
                              'collected':len(ids)>=self.cfg['min_pixels_window']})
        if not indices:
            if self.trace_enabled:
                self.trace['candidates'].append({'start':float(start),'steps':steps,'fit':None,
                                                 'reason':'no populated windows'})
            return None,windows
        # Windows have disjoint y ranges. Reverse to restore row order without sorting.
        ids=np.concatenate(indices[::-1])
        fit,reason,details=fit_supported_rows(xx[ids],yy[ids],self.cfg,self.validation,h)
        if fit is None:
            self.rejections.append({'reason':reason,**details})
        else:
            self.accepted_fits+=1
        if self.trace_enabled:
            self.trace['candidates'].append({'start':float(start),'steps':steps,'fit':fit,
                'raw_coefficients':fit.coefficients.copy() if fit else None,
                'reason':reason,'support':details})
        return fit,windows

    def detect(self,binary):
        self.rejections=[]
        self.accepted_fits=0
        if self.trace_enabled:
            self.trace={'candidates':[],'selected':{}}
        self._coordinates=binary.nonzero()
        try:
            result=super().detect(binary)
        finally:
            self._coordinates=None
        if result.status=='FAILED' and self.rejections and self.accepted_fits==0:
            result.status='REJECTED_SUPPORT'
            result.reason='; '.join(sorted(set(r['reason'] for r in self.rejections)))
        return result
