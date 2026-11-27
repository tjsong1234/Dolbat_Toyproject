"""Detect crosswalk/stop markings BEFORE Top-hat destroys their original width.

The suppression mask follows individual paint shapes; it never blacks out a whole
crosswalk rectangle, so neighboring lane boundaries can survive.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class SuppressionResult:
    binary: np.ndarray
    raw_paint: np.ndarray
    crosswalk: np.ndarray
    horizontal: np.ndarray
    removed: np.ndarray
    stripe_count: int


def vertical_run_lengths(binary):
    """Length of the actual contiguous paint run in each column, zero off paint.

    A vertical lane passing through a thin stop line has a long run and is protected.
    This is a directional pixel thickness, not a physical distance measurement.
    """
    paint=binary!=0
    rows=np.arange(binary.shape[0],dtype=np.int32)[:,None]
    before=np.maximum.accumulate(np.where(paint,-1,rows),axis=0)
    after=np.minimum.accumulate(np.where(paint,binary.shape[0],rows)[::-1],axis=0)[::-1]
    return np.where(paint,after-before-1,0)


def short_vertical_runs(paint,max_length):
    """Keep paint runs of at most max_length, without full int32 run images.

    Erosion marks starts of runs with max_length+1 pixels. Dilation with the
    opposing anchor restores those runs exactly, including even kernel sizes.
    Explicit zero borders preserve the finite-image run definition.
    """
    length=int(np.floor(max_length))+1
    if length<=0:return np.zeros_like(paint)
    kernel=np.ones((length,1),np.uint8)
    starts=cv2.erode(paint,kernel,anchor=(0,0),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    long_runs=cv2.dilate(starts,kernel,anchor=(0,length-1),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    return cv2.bitwise_and(paint,cv2.bitwise_not(long_runs))


def thickness_local_horizontal(paint,cfg):
    """Reconstruct recognized horizontal paint ONLY inside short vertical runs.

    No fitted line extension or deletion-mask dilation is used. Component lookup
    restores slanted ends within actual paint, without cutting a crossing long lane.
    """
    length=cfg.get('horizontal_length_px',95)
    seed=cv2.morphologyEx(paint,cv2.MORPH_OPEN,
                         np.ones((cfg.get('horizontal_thickness_px',2),length),np.uint8))
    thin=short_vertical_runs(paint,cfg.get('horizontal_max_vertical_thickness_px',28))
    n,labels,stats,_=cv2.connectedComponentsWithStats(thin)
    lookup=np.zeros(n,dtype=np.uint8)
    touched=np.unique(labels[seed!=0]);touched=touched[touched!=0]
    for label in touched:
        # A shallow part of a curved lane can contain a horizontal opening seed.
        # Do not reconstruct that seed through an entire bending paint component.
        # Test the centerline of the ACTUAL short-run component, without extending
        # a fitted line or dilating the deletion mask into neighboring paint.
        if 'horizontal_max_center_slope' in cfg:
            x,y,w,h,_=stats[label]
            if w<cfg.get('horizontal_min_component_span_px',length//2):continue
            yy,xx=np.nonzero(labels[y:y+h,x:x+w]==label)
            count=np.bincount(xx,minlength=w)
            sums=np.bincount(xx,weights=yy,minlength=w)
            xs=np.flatnonzero(count);centers=sums[xs]/count[xs]
            co=np.polyfit(xs,centers,1)
            rms=float(np.sqrt(np.mean((centers-np.polyval(co,xs))**2)))
            if abs(co[0])>cfg['horizontal_max_center_slope']:continue
            if rms>cfg.get('horizontal_max_center_rms_px',4):continue
        lookup[label]=255
    return lookup[labels]


def suppress_markings(bev, candidate, cfg, hsv=None):
    # Geometry preprocessing has already converted this exact BEV image.
    if hsv is None:hsv=cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    neutral=cfg.get('raw_paint_min_channel',175)
    paint=cv2.bitwise_and(cv2.inRange(bev,(neutral,neutral,neutral),(255,255,255)),
                          cv2.inRange(hsv,(0,0,0),(179,cfg.get('raw_paint_max_saturation',65),255)))
    length = cfg.get('horizontal_length_px',95)
    thickness = cfg.get('horizontal_thickness_px',2)
    local=cfg.get('mode','legacy')=='thickness_local'
    if local:
        horizontal=thickness_local_horizontal(paint,cfg)
    else:
        horizontal = cv2.morphologyEx(paint,cv2.MORPH_OPEN,np.ones((thickness,length),np.uint8))
        pad = cfg.get('horizontal_padding_px',5)
        extended=horizontal.copy()
        hn,hl,hs,_=cv2.connectedComponentsWithStats(horizontal)
        for label in range(1,hn):
            x0,y0,w,h,area=hs[label]
            if w<length or w<4*h:
                continue
            yy,xx=np.nonzero(hl[y0:y0+h,x0:x0+w]==label)
            count=np.bincount(xx,minlength=w)
            ys=np.bincount(xx,weights=yy,minlength=w)
            valid=count>0;columns=np.flatnonzero(valid);centers=ys[valid]/count[valid]
            co=np.polyfit(columns,centers,1)
            if abs(co[0])>.35:
                continue
            start=max(0,int(x0)-length);end=min(paint.shape[1]-1,int(x0+w-1)+length)
            xs=np.arange(start,end+1)
            points=np.column_stack((xs,np.polyval(co,xs-x0)+y0)).astype(np.int32)
            stroke=max(3,int(np.ceil(np.percentile(count[valid],90)))+pad)
            cv2.polylines(extended,[points],False,255,stroke)
        horizontal=cv2.dilate(extended,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(pad,pad)))

    # Work on raw paint, rather than narrow Top-hat remnants at the stripe corners.
    n,labels,stats,centroids = cv2.connectedComponentsWithStats(paint)
    distance = cv2.distanceTransform(paint,cv2.DIST_L2,3)
    stripes=[]
    min_width,max_width = cfg.get('stripe_width_range_px',[18,130])
    for label in range(1,n):
        x,y,w,h,area = stats[label]
        if (area >= cfg.get('stripe_min_area',600) and h >= cfg.get('stripe_min_height_px',55)
            and min_width <= w <= max_width
            and h/w >= cfg.get('stripe_min_height_width_ratio',1.3)
            and area/(w*h) >= cfg.get('stripe_min_fill',.5)):
            # Only query possible stripes; a full-image indexed maximum was the
            # dominant CPU cost, including hundreds of thousands of background pixels.
            local_labels=labels[y:y+h,x:x+w]
            maximum=distance[y:y+h,x:x+w].max(where=local_labels==label,initial=0)
            if maximum >= cfg.get('stripe_min_half_thickness_px',9):
                stripes.append(label)
    # Group stripes with matching vertical extents and widths. Dashed lane segments
    # align end-to-end in y, whereas a crosswalk contains several side-by-side bars.
    neighbors={i:set() for i in stripes}
    for j,a in enumerate(stripes):
        xa,ya,wa,ha,_ = stats[a]
        for b in stripes[j+1:]:
            xb,yb,wb,hb,_ = stats[b]
            overlap=max(0,min(ya+ha,yb+hb)-max(ya,yb))/min(ha,hb)
            if (overlap >= cfg.get('stripe_min_vertical_overlap',.65)
                and max(ha,hb)/min(ha,hb) <= cfg.get('stripe_max_height_ratio',1.7)
                and max(wa,wb)/min(wa,wb) <= cfg.get('stripe_max_width_ratio',2.2)):
                neighbors[a].add(b);neighbors[b].add(a)
    accepted=set();visited=set()
    for start in stripes:
        if start in visited:
            continue
        group=set();pending=[start]
        while pending:
            label=pending.pop()
            if label in group:
                continue
            group.add(label);visited.add(label);pending.extend(neighbors[label]-group)
        if len(group) < cfg.get('stripe_group_min_count',3):
            continue
        centers=sorted(centroids[label,0] for label in group)
        gaps=np.diff(centers)
        if gaps.mean() > 0 and gaps.std()/gaps.mean() <= cfg.get('stripe_max_spacing_cv',.45):
            accepted.update(group)
    lookup=np.zeros(n,dtype=np.uint8)
    if accepted:
        lookup[list(accepted)]=255
    crosswalk=lookup[labels]
    pad=cfg.get('stripe_padding_px',7)
    if not local:
        crosswalk=cv2.dilate(crosswalk,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(pad,pad)))
    excluded=cv2.bitwise_or(crosswalk,horizontal)
    binary=cv2.bitwise_and(candidate,cv2.bitwise_not(excluded))
    return SuppressionResult(binary,paint,crosswalk,horizontal,cv2.bitwise_and(candidate,excluded),len(accepted))
