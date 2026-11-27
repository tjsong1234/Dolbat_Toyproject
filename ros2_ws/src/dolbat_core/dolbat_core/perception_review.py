"""Candidate masks and visual evidence for the saved-ROI sliding-window review."""
import cv2
import numpy as np


def candidate_mask(bev, legacy, method, cfg):
    if method == 'legacy':
        return legacy
    hsv = cv2.cvtColor(bev,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bev,cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(hsv,(0,0,cfg.get('white_min_value',165)),
                        (179,cfg.get('white_max_saturation',85),255))
    k = cfg.get('tophat_kernel_px',35)
    top = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    ridge = cv2.inRange(top,cfg.get('tophat_min_value',25),255)
    mask = cv2.bitwise_and(white,ridge)
    if method == 'tophat':
        return mask
    if method != 'refined':
        raise ValueError(f'Unknown mask method: {method}')
    # Filter BEFORE road adjacency, so broad zebra stripes do not turn into thin hollow outlines.
    dist = cv2.distanceTransform(mask,cv2.DIST_L2,3)
    seed = np.uint8(dist >= cfg.get('refined_min_half_thickness_px',3.5))*255
    mask = cv2.bitwise_and(mask,cv2.dilate(seed,np.ones((7,7),np.uint8)))
    # Constrain components by stroke geometry, without requiring a global vertical orientation.
    n,labels,stats,_ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for label in range(1,n):
        x,y,w,h,area = stats[label]
        if area < cfg.get('min_component_area',70) or h < 10 or w > 3.5*h:
            continue
        clean[labels==label] = 255
    return clean


def draw_lanes(image, lanes, inverse=None):
    out = image.copy()
    for fit,color in ((lanes.left,(255,170,30)),(lanes.right,(20,225,255))):
        if fit is None:
            continue
        y = np.linspace(fit.y_min,fit.y_max,120)
        xy = np.column_stack((fit.x(y),y)).astype(np.float32)
        if inverse is not None:
            xy = cv2.perspectiveTransform(xy.reshape(-1,1,2),inverse).reshape(-1,2)
        valid = np.isfinite(xy).all(axis=1) & (np.abs(xy).max(axis=1)<100000)
        xy = xy[valid].astype(np.int32)
        if fit.inferred:
            for j in range(0,len(xy)-3,8):
                cv2.polylines(out,[xy[j:j+4]],False,color,2,cv2.LINE_AA)
        elif len(xy)>1:
            cv2.polylines(out,[xy],False,color,3,cv2.LINE_AA)
    return out


def tile(image, title, size=(320,240)):
    if image.ndim == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    out = cv2.resize(image,size)
    cv2.rectangle(out,(0,0),(size[0],25),(22,24,28),-1)
    cv2.putText(out,title,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(235,240,245),1,cv2.LINE_AA)
    return out
