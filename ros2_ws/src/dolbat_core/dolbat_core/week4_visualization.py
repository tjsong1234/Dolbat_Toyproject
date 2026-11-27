"""Fixed-scale rear-axle plot and four synchronized course-required views."""
import math
from functools import lru_cache
import cv2
import numpy as np
from .final_visualization import boundary_curves, green_overlay
from .week3_visualization import polyline

BG=(24,28,35)
WHITE=(232,239,242)
GREEN=(75,225,100)
MAGENTA=(225,105,245)


def text(im,value,pos,scale=.48,color=WHITE,thickness=1):
    cv2.putText(im,str(value),pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


@lru_cache(maxsize=16)
def _picture_background(title,size):
    w,h=size;out=np.full((h,w,3),BG,np.uint8)
    text(out,title,(12,22),.50)
    out.setflags(write=False)
    return out


def picture(image,title,size=(640,480)):
    w,h=size;out=_picture_background(title,tuple(size)).copy()
    factor=min(w/image.shape[1],(h-32)/image.shape[0])
    scaled=cv2.resize(image,(round(image.shape[1]*factor),round(image.shape[0]*factor)))
    x=(w-scaled.shape[1])//2;y=32+(h-32-scaled.shape[0])//2
    out[y:y+scaled.shape[0],x:x+scaled.shape[1]]=scaled
    return out


def plot_transform(settings,size=(640,480)):
    w,h=size;xmin=float(settings['x_min_m']);xmax=float(settings['x_max_m'])
    ylim=float(settings['y_limit_m']);tick=float(settings['tick_m'])
    if not np.isfinite([xmin,xmax,ylim,tick]).all() or xmax<=xmin or min(ylim,tick)<=0:
        raise ValueError('Invalid fixed plot axes')
    scale=min((w-100)/(2*ylim),(h-80)/(xmax-xmin))
    top=35+((h-80)-(xmax-xmin)*scale)/2
    return dict(x_min_m=xmin,x_max_m=xmax,y_limit_m=ylim,tick_m=tick,
                scale_px_per_m=scale,origin_u=w/2,origin_v=top+xmax*scale,top=top)


def to_pixels(points,transform):
    points=np.asarray(points,dtype=float)
    return np.column_stack((transform['origin_u']-points[:,1]*transform['scale_px_per_m'],
                            transform['origin_v']-points[:,0]*transform['scale_px_per_m']))


@lru_cache(maxsize=16)
def _metric_background(xmin,xmax,ylim,tick,lookahead,size):
    w,h=size;out=np.full((h,w,3),BG,np.uint8)
    tr=plot_transform(dict(x_min_m=xmin,x_max_m=xmax,y_limit_m=ylim,tick_m=tick),size)
    xmin,xmax,ylim,tick=[tr[k] for k in ('x_min_m','x_max_m','y_limit_m','tick_m')]
    pix=lambda p:np.rint(to_pixels(p,tr)).astype(int)
    for x in np.arange(math.ceil(xmin/tick)*tick,xmax+1e-9,tick):
        p=pix([[x,-ylim],[x,ylim]]);cv2.line(out,tuple(p[0]),tuple(p[1]),(55,60,67),1)
        text(out,f'{x:g}',(12,int(p[0,1])+4),.34,(185,196,207))
    for y in np.arange(-ylim,ylim+1e-9,tick):
        p=pix([[xmin,y],[xmax,y]]);cv2.line(out,tuple(p[0]),tuple(p[1]),(55,60,67),1)
        text(out,f'{y:g}',(int(p[0,0])-12,h-9),.34,(185,196,207))
    origin=tuple(pix([[0,0]])[0]);radius=round(lookahead*tr['scale_px_per_m'])
    cv2.circle(out,origin,radius,(80,150,230),1,cv2.LINE_AA)
    out.setflags(write=False)
    return out


def _observed_discs(out,xy):
    """Rasterize radius-one filled LINE_8 circles as their exact five pixels."""
    h,w=out.shape[:2]
    xy=xy[(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=32)&(xy[:,1]<h)]
    if not len(xy):return
    points=(xy[:,None,:]+np.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]])).reshape(-1,2)
    points=points[(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)]
    out[points[:,1],points[:,0]]=(175,180,185)


def metric_plot(plan,path_rear,result,translation,cfg,size=(640,480)):
    w,h=size;tr=plot_transform(cfg['visualization'],size)
    out=_metric_background(*(tr[k] for k in ('x_min_m','x_max_m','y_limit_m','tick_m')),
                           float(cfg['lookahead_m']),tuple(size)).copy()
    pix=lambda p:np.rint(to_pixels(p,tr)).astype(int)
    origin=tuple(pix([[0,0]])[0])
    for side,points,color,inferred in boundary_curves(plan):
        polyline(out,to_pixels(np.asarray(points)/100+translation,tr),color,2,False)
    for side in ('left','right'):
        boundary=getattr(plan,side)
        if boundary is not None:
            xy=pix(boundary.row_centers_cm/100+translation)
            _observed_discs(out,xy)
    for points in getattr(plan,'partial_points_cm',{}).values():
        xy=pix(points[::max(1,len(points)//180)]/100+translation)
        xy=xy[(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=32)&(xy[:,1]<h)]
        out[xy[:,1],xy[:,0]]=(175,110,220)
    if len(path_rear):polyline(out,to_pixels(path_rear,tr),GREEN,2,False)
    cv2.circle(out,origin,4,WHITE,-1,cv2.LINE_AA)
    cv2.arrowedLine(out,origin,tuple(pix([[.35,0]])[0]),WHITE,2,cv2.LINE_AA,tipLength=.25)
    anchor=tuple(pix([[cfg['rear_to_image_bottom_m'],0]])[0])
    cv2.drawMarker(out,anchor,(180,185,190),cv2.MARKER_CROSS,10,1)
    text(out,'Image bottom',(anchor[0]+8,anchor[1]+4),.32,(180,185,190))
    if result.valid:
        target=tuple(pix([result.target_m])[0]);cv2.line(out,origin,target,MAGENTA,1,cv2.LINE_AA)
        cv2.circle(out,target,6,MAGENTA,-1,cv2.LINE_AA);text(out,'LP',(target[0]+9,target[1]-7),.46,MAGENTA)
    text(out,'3 | Rear axle: X forward / Y left (m); FIXED axes',(10,21),.45)
    text(out,'Rear axle',(origin[0]+10,origin[1]+4),.34)
    return out


def camera_overlay(image,plan,mapping,inverse,result,translation):
    out=green_overlay(image,plan,mapping,inverse)
    if result.valid:
        bev_cm=(np.array(result.target_m)-translation)*100
        uv=mapping.cm_to_pixels(bev_cm[None,:]).astype(np.float32)
        projected=cv2.perspectiveTransform(uv.reshape(1,1,2),inverse).reshape(2)
        if np.isfinite(projected).all() and 0<=projected[0]<out.shape[1] and 0<=projected[1]<out.shape[0]:
            xy=tuple(np.rint(projected).astype(int));cv2.circle(out,xy,6,MAGENTA,-1,cv2.LINE_AA)
            text(out,'LP',(xy[0]+9,xy[1]-6),.45,MAGENTA)
    return out


@lru_cache(maxsize=16)
def _control_background(total,fps,lookahead,wheelbase):
    out=np.full((480,640,3),BG,np.uint8)
    text(out,'4 | Pure Pursuit / live terminal values',(12,23),.54)
    text(out,f'LD {lookahead:.2f} m | Wheelbase {wheelbase:.2f} m',(18,273),.49)
    left,right,top,bottom=54,615,329,443
    duration=total/fps
    for degree in (-20,0,20):
        y=round(bottom-(degree+25)/50*(bottom-top));cv2.line(out,(left,y),(right,y),(60,66,72),1)
        text(out,str(degree),(13,y+4),.34)
    for second in range(0,math.ceil(duration)+1,5):
        x=round(left+second/duration*(right-left))
        if x<=right:cv2.line(out,(x,top),(x,bottom),(60,66,72),1);text(out,str(second),(x-4,461),.33)
    text(out,'deg',(9,322),.32);text(out,'seconds',(565,477),.30)
    out.setflags(write=False)
    return out


def control_panel(result,index,total,fps,pipeline_ms,cfg,history):
    out=_control_background(total,fps,float(cfg['lookahead_m']),float(cfg['wheelbase_m'])).copy()
    text(out,f'Frame {index+1}/{total} | {index/fps:.2f} s',(15,57),.52)
    if result.valid:
        text(out,f'{result.steering_deg:+.2f} deg',(18,120),1.55,GREEN,2)
        direction='LEFT (+)' if result.steering_deg>1e-8 else 'RIGHT (-)' if result.steering_deg< -1e-8 else 'STRAIGHT'
        lines=[f'{direction} | {result.steering_rad:+.5f} rad',
               f'LP: X={result.target_m[0]:.4f}, Y={result.target_m[1]:+.4f} m',
               f'Curvature: {result.curvature_per_m:+.5f} /m',
               f'LP circle residual: {result.circle_residual_m:.2e} m']
    else:lines=['UNAVAILABLE: '+result.status,'No steering command / no fabricated target']
    for j,line in enumerate(lines):text(out,line,(18,156+26*j),.50)
    text(out,f'Processing throughput (rolling): {1000/max(.001,pipeline_ms):.1f} FPS',(18,299),.44)
    # The time and steering axes stay constant throughout the recording.
    left,right,top,bottom=54,615,329,443
    duration=total/fps
    if len(history):
        values=np.asarray(history,dtype=float)
        points=np.column_stack((left+(np.arange(len(values))/fps)/duration*(right-left),
                                bottom-(values+25)/50*(bottom-top)))
        # None is the only unavailable-history marker in the existing API.
        missing=np.fromiter((value is None for value in history),dtype=bool,count=len(history))
        starts=np.flatnonzero(~missing&~np.r_[False,~missing[:-1]])
        ends=np.flatnonzero(~missing&~np.r_[~missing[1:],False])+1
        for a,b in zip(starts,ends):
            if b-a>1:polyline(out,points[a:b],GREEN,1,False)
    return out


@lru_cache(maxsize=1)
def _footer_background():
    out=np.full((64,1280,3),(21,36,28),np.uint8)
    text(out,'Recorded-video calculation | assumed camera centreline / heading alignment | positive LEFT | no vehicle actuation',(12,51),.46)
    out.setflags(write=False)
    return out


def compose(frame,preprocessed,lanes,plan,mapping,inverse,path_rear,result,translation,cfg,index,total,fps,ms,history):
    metric=metric_plot(plan,path_rear,result,translation,cfg)
    control=control_panel(result,index,total,fps,ms,cfg,history)
    final=np.empty((1024,1280,3),np.uint8)
    final[:480,:640]=picture(frame,'1 | Original recording')
    final[:480,640:]=picture(lanes.debug,'2 | BEV / Sliding Window')
    final[480:960,:640]=metric;final[480:960,640:]=control
    footer=_footer_background().copy()
    text(footer,f'Frame {index+1}/{total} | perception {lanes.status} | path {plan.status} | control {result.status}',(12,24),.47)
    final[960:]=footer
    overlay=camera_overlay(frame,plan,mapping,inverse,result,translation)
    greenfooter=footer.copy()
    angle=f'{result.steering_deg:+.3f} deg' if result.valid else result.status
    text(greenfooter,'Pure Pursuit: '+angle,(870,24),.48,GREEN)
    green=np.empty((544,1280,3),np.uint8)
    green[:480,:640]=picture(overlay,'Green path + LP on original')
    green[:480,640:]=metric;green[480:]=greenfooter
    return final,green,control,metric
