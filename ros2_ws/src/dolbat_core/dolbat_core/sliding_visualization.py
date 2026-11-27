"""Educational views built from the detector's actual observations, not a replay."""
from pathlib import Path
import cv2
import numpy as np
from .perception_review import draw_lanes, tile

COLORS={'left':(255,170,30),'right':(20,225,255)}


def save_png(path,image):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ok,data=cv2.imencode('.png',image)
    if not ok:
        raise OSError(f'PNG encoding failed: {path}')
    path.write_bytes(data.tobytes())


def annotate(image,title,lines=()):
    if image.ndim==2:
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    w=image.shape[1]
    top=np.full((38,w,3),(24,28,35),np.uint8)
    cv2.putText(top,title,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,244,250),1,cv2.LINE_AA)
    bottom=np.full((max(1,len(lines))*26+10,w,3),(24,28,35),np.uint8)
    for i,line in enumerate(lines):
        cv2.putText(bottom,line,(10,23+i*26),cv2.FONT_HERSHEY_SIMPLEX,.47,(225,231,240),1,cv2.LINE_AA)
    return np.vstack((top,image,bottom))


def selected_traces(detector,result):
    if not result.usable:
        return {}
    return {s:t for s,t in detector.trace.get('selected',{}).items() if t is not None}


def histogram_view(result,detector,selected=True):
    hist=result.histogram;smooth=np.convolve(hist,np.ones(15)/15,'same')
    out=np.full((480,640,3),(22,26,33),np.uint8)
    x0,y0,pw,ph=55,405,565,335
    maximum=max(10.,float(hist.max())*1.1)
    for i in range(5):
        y=int(y0-ph*i/4)
        cv2.line(out,(x0,y),(x0+pw,y),(58,60,66),1)
        cv2.putText(out,f'{maximum*i/4:.0f}',(8,y+5),0,.4,(200,200,200),1)
    for values,color in ((hist,(130,135,145)),(smooth,(240,240,240))):
        points=np.column_stack((x0+np.arange(len(hist))*pw/(len(hist)-1),y0-values/maximum*ph)).astype(np.int32)
        cv2.polylines(out,[points],False,color,1,cv2.LINE_AA)
    # All seeds include histogram peaks plus previous-frame seeds, when available.
    for t in detector.trace.get('candidates',[]):
        xx=int(x0+t['start']*pw/(len(hist)-1))
        if x0<=xx<=x0+pw:
            cv2.line(out,(xx,y0),(xx,y0-12),(140,110,210),2)
    if selected:
        for side,t in selected_traces(detector,result).items():
            xx=int(x0+t['start']*pw/(len(hist)-1))
            if x0<=xx<=x0+pw:
                cv2.line(out,(xx,y0),(xx,y0-ph),COLORS[side],2)
                cv2.putText(out,f'{side}: {t["start"]:.1f}',(max(60,min(xx-30,480)),45),0,.48,COLORS[side],1)
    for x in (0,160,320,480,639):
        cv2.putText(out,str(x),(int(x0+x*pw/639)-10,y0+23),0,.4,(220,220,220),1)
    cv2.putText(out,'x (BEV pixels)',(265,468),0,.5,(230,230,230),1)
    cv2.putText(out,'Count from y=168..479 | gray=raw white=15px mean',(60,20),0,.45,(225,225,225),1)
    return out


def window_view(mask,detector,result,count=12):
    out=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    traces=selected_traces(detector,result)
    for side,t in traces.items():
        color=COLORS[side]
        for step in t['steps'][:count]:
            x1,y1,x2,y2=step['bounds']
            px=step['pixels'].astype(int)
            if step['collected'] and len(px):
                out[px[:,1],px[:,0]]=color
            active=step['index']==count
            cv2.rectangle(out,(x1,y1),(x2,y2),color if active else (65,135,65),2 if active else 1)
            y=(y1+y2)//2
            cv2.circle(out,(int(step['center_before']),y),3,(240,80,210),-1)
            if step['median'] is not None:
                cv2.circle(out,(int(step['median']),y),4,color,-1)
                cv2.arrowedLine(out,(int(step['center_before']),y),(int(step['median']),y),color,1,tipLength=.2)
            if -15<x1<640:
                cv2.putText(out,str(step['index']),(max(0,x1+3),y1+15),0,.4,color,1)
    if not traces:
        # Failed hypotheses are shown only as rejected, never as selected lanes.
        for t in detector.trace.get('candidates',[])[:3]:
            for step in t['steps'][:count]:
                x1,y1,x2,y2=step['bounds'];cv2.rectangle(out,(x1,y1),(x2,y2),(60,60,150),1)
        cv2.putText(out,'NO ACCEPTED LANE - rejected search hypotheses',(8,28),0,.5,(90,90,255),1)
    return out


def pixels_view(mask,result):
    out=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)//3
    for side,lane in (('left',result.left),('right',result.right)):
        if lane is not None and not lane.inferred and len(lane.pixels):
            px=lane.pixels.astype(int);out[px[:,1],px[:,0]]=COLORS[side]
    return out


def dashboard(processed,mask,result,detector,inverse,frame,n,fps,ms):
    original=draw_lanes(processed.original,result,inverse)
    cv2.polylines(original,[processed.source_points.astype(np.int32)],True,(130,200,130),1)
    panels=[tile(original,'1 Original / fixed ROI / lane projection',(480,360)),
            tile(draw_lanes(processed.bev,result),'2 BEV / fitted boundaries',(480,360)),
            tile(mask,'3 Binary mask (week 1 final preprocessing)',(480,360)),
            tile(histogram_view(result,detector),'4 Histogram / candidate and selected seeds',(480,360)),
            tile(window_view(mask,detector,result),'5 Sliding windows (bottom to top)',(480,360)),
            tile(draw_lanes(pixels_view(mask,result),result),'6 Inlier pixels / polynomial fit',(480,360))]
    footer=np.full((80,1440,3),(22,26,32),np.uint8)
    color=(100,230,130) if result.status=='DETECTED' else (70,195,255) if result.usable else (90,90,255)
    cv2.putText(footer,f'Frame {frame+1}/{n} (index {frame}) | t={frame/fps:.2f}s | {result.status} | perception {1000/max(ms,.001):.1f} FPS',
                (12,25),0,.65,color,1,cv2.LINE_AA)
    cv2.putText(footer,'Blue=left / yellow=right | SOLID=observed + smoothed | DASHED=inferred | Status is NOT accuracy',
                (12,50),0,.55,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(footer,result.reason[:155],(12,72),0,.45,(190,195,205),1,cv2.LINE_AA)
    return np.vstack((np.hstack(panels[:3]),np.hstack(panels[3:]),footer))
