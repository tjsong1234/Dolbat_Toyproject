"""Readable week-3 overlays: current evidence, partial evidence, and green path."""
from pathlib import Path
import cv2
import numpy as np
from .week3_visualization import polyline,tile_keep,LEFT,RIGHT,PATH

DEFAULT_METRIC_AXES={'x_min':-20.,'x_max':380.,'y_limit':200.,'tick_cm':50.}


def set_metric_axes(plan,settings=None):
    """Attach display settings only; never alter model or path coordinates."""
    plan.metric_axes={**DEFAULT_METRIC_AXES,**(settings or {})}
    metric_view_bounds(plan)  # Validate before drawing any output.
    return plan


def attach_partials(plan,lanes,mapping):
    plan.partial_points_cm={}
    for side in ('left','right'):
        fit=getattr(lanes,'partial_'+side,None)
        if fit is not None:plan.partial_points_cm[side]=mapping.pixels_to_cm(fit.pixels)


def boundary_curves(plan):
    """Return continuous boundary models, keeping inferred sides explicit.

    A yellow-guided full model has priority over a short observed fit. Its samples
    remain a model: observed scatter is read separately and is never extended.
    """
    curves=[]
    for side,color in (('left',LEFT),('right',RIGHT)):
        boundary=getattr(plan,side)
        full=getattr(plan,'full_model_'+side+'_cm',None)
        if full is not None and len(full):
            mask=getattr(plan,'full_model_'+side+'_inferred_mask',None)
            inferred=bool(np.any(mask)) if mask is not None else boundary is None
            curves.append((side,np.asarray(full),color,inferred))
            continue
        if boundary is not None:
            xx=np.linspace(*boundary.interval_cm,180)
            curves.append((side,np.column_stack((xx,boundary.y(xx))),color,False))
            continue
        points=getattr(plan,'boundary_pair_'+side+'_cm',None)
        if points is None or not len(points):
            points=plan.inferred_boundary_cm
        if points is not None and len(points):curves.append((side,np.asarray(points),color,True))
    return curves


def boundary_visibility(image_shape,projected_points,thickness=2):
    """Report actual polyline visibility without clamping an offscreen model.

    Rasterization includes segments crossing the camera even if their endpoints
    are outside it. The returned side only locates an explanatory UI indicator.
    """
    h,w=image_shape[:2];mask=np.zeros((h,w),np.uint8)
    points=np.asarray(projected_points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2:
        return {'visible':False,'visible_pixels':0,'side':'left'}
    good=np.isfinite(points).all(axis=1)&(np.max(abs(points),axis=1)<100000)
    points=points[good]
    if len(points)>1:polyline(mask,points,255,thickness,False)
    count=int(np.count_nonzero(mask))
    side='left'
    if len(points):
        if np.all(points[:,0]>=w):side='right'
        elif np.all(points[:,1]<0):side='top'
        elif np.all(points[:,1]>=h):side='bottom'
    return {'visible':bool(count),'visible_pixels':count,'side':side}


def metric_view_bounds(plan):
    """Fixed cm axes: their bounds must not depend on this frame's observations."""
    result={**DEFAULT_METRIC_AXES,**getattr(plan,'metric_axes',{})}
    result={k:float(result[k]) for k in DEFAULT_METRIC_AXES}
    if (not np.isfinite(list(result.values())).all() or result['x_max']<=result['x_min']
            or result['y_limit']<=0 or result['tick_cm']<=0):
        raise ValueError('Finite, ordered metric axes and positive limits/tick spacing required')
    return result


def metric_view_transform(plan,size=(480,360)):
    """One fixed affine mapping shared by grid, observed points and models."""
    w,h=size;bounds=metric_view_bounds(plan)
    span=bounds['x_max']-bounds['x_min']
    scale=min((w-75)/(2*bounds['y_limit']),(h-65)/span)
    top=25+((h-65)-span*scale)/2
    return {**bounds,'scale':scale,'cx':w/2,'top':top,'bottom':top+span*scale,
        'origin_u':w/2,'origin_v':top+bounds['x_max']*scale}


def _draw_offscreen_indicator(image,visibility):
    h,w=image.shape[:2];side=visibility['side']
    if side=='right':start=(w-100,38);end=(w-14,38);text=(max(10,w-255),62)
    elif side=='top':start=(34,88);end=(34,14);text=(48,30)
    elif side=='bottom':start=(34,h-88);end=(34,h-14);text=(48,h-30)
    else:start=(100,38);end=(14,38);text=(12,62)
    cv2.arrowedLine(image,start,end,LEFT,2,cv2.LINE_AA,tipLength=.2)
    cv2.putText(image,'Left model outside camera',text,0,.45,LEFT,1,cv2.LINE_AA)


def green_overlay(image,plan,mapping,inverse=None):
    out=image.copy()
    def project(points):
        uv=mapping.cm_to_pixels(points).astype(np.float32)
        return cv2.perspectiveTransform(uv.reshape(-1,1,2),inverse).reshape(-1,2) if inverse is not None else uv
    corridor=None
    pair_left=getattr(plan,'full_model_left_cm',None)
    pair_right=getattr(plan,'full_model_right_cm',None)
    if pair_left is None or pair_right is None or not len(pair_left) or not len(pair_right):
        pair_left=getattr(plan,'boundary_pair_left_cm',None)
        pair_right=getattr(plan,'boundary_pair_right_cm',None)
    if plan.usable and pair_left is not None and pair_right is not None and len(pair_left) and len(pair_right):
        corridor=np.vstack((pair_left,pair_right[::-1]))
    elif plan.left is not None and plan.right is not None and plan.usable:
        x=np.linspace(plan.path_cm[0,0],plan.path_cm[-1,0],100)
        corridor=np.vstack((np.column_stack((x,plan.left.y(x))),np.column_stack((x,plan.right.y(x)))[::-1]))
    elif plan.usable and plan.inferred_boundary_cm is not None and 'TRUNCATED' not in plan.status:
        boundary=plan.left if plan.left is not None else plan.right
        xx=np.linspace(*boundary.interval_cm,len(plan.inferred_boundary_cm))
        corridor=np.vstack((np.column_stack((xx,boundary.y(xx))),plan.inferred_boundary_cm[::-1]))
    if corridor is not None:
        pts=project(corridor)
        if np.isfinite(pts).all() and np.max(abs(pts))<100000:
            layer=out.copy();cv2.fillPoly(layer,[np.rint(pts).astype(np.int32)],(60,200,80))
            out=cv2.addWeighted(layer,.12 if plan.status=='OBSERVED_PAIR' else .07,out,.88 if plan.status=='OBSERVED_PAIR' else .93,0)
    left_visibility=None;projected_models=[]
    for side,points,color,inferred in boundary_curves(plan):
        # Completed boundary geometry is continuous. Inference is identified in
        # labels, not by introducing visible holes into the completed model.
        projected=project(points)
        projected_models.append((projected,color))
        if side=='left' and inverse is not None:left_visibility=boundary_visibility(out.shape,projected)
    # A pointwise center exists even when one boundary is inferred. Its display
    # must not inherit the dashed style used to identify an inferred boundary.
    for side,points in getattr(plan,'partial_points_cm',{}).items():
        uv=project(points[::max(1,len(points)//250)])
        uv=uv[np.isfinite(uv).all(axis=1)].astype(int)
        valid=(uv[:,0]>=0)&(uv[:,0]<out.shape[1])&(uv[:,1]>=0)&(uv[:,1]<out.shape[0])
        out[uv[valid,1],uv[valid,0]]=(180,120,255)
    # Measured fragments must not overpaint holes into the continuous model.
    for projected,color in projected_models:polyline(out,projected,color,2,False)
    if plan.usable:polyline(out,project(plan.path_cm),PATH,3,False)
    if left_visibility is not None and not left_visibility['visible']:_draw_offscreen_indicator(out,left_visibility)
    return out


def green_frame(image,plan,mapping,inverse,index,total,fps,show_model=False):
    overlay=green_overlay(image,plan,mapping,inverse)
    if show_model:overlay=np.hstack((overlay,metric_with_partials(plan,2,(480,image.shape[0]))))
    footer=np.full((56,overlay.shape[1],3),(22,29,24),np.uint8)
    completed=getattr(plan,'full_model_left_cm',None)
    label=('Joint B-spline center / width' if getattr(plan,'joint_fit_accepted',False) else
           'Joint fit fallback' if plan.status.startswith('FALLBACK') else
           'Blue follows current yellow' if getattr(plan,'yellow_guided',False) and plan.usable
           else ('Two observed boundaries' if plan.status=='OBSERVED_PAIR' else ('One-side estimate' if plan.usable else 'Path unavailable')))
    cv2.putText(footer,f'{index+1}/{total} | {index/fps:.2f}s | {label}',(10,21),0,.50,(235,244,237),1,cv2.LINE_AA)
    cv2.putText(footer,'Blue: model | green: center | gray dots: observed paint | pink: partial',(10,43),0,.37,(210,228,215),1,cv2.LINE_AA)
    return np.vstack((overlay,footer))


def metric_with_partials(plan,stage=2,size=(480,360)):
    # Keep viewport scale, origin and grid fixed while the geometry moves.
    w,h=size;out=np.full((h,w,3),(24,28,35),np.uint8);bounds=metric_view_transform(plan,size)
    xmin,xmax,limit=bounds['x_min'],bounds['x_max'],bounds['y_limit']
    scale,cx,top,bottom=(bounds[k] for k in ('scale','cx','top','bottom'))
    def pixels(points):
        points=np.asarray(points)
        return np.column_stack((cx-points[:,1]*scale,top+(xmax-points[:,0])*scale))
    tick=bounds['tick_cm']
    for x in np.arange(np.ceil(xmin/tick)*tick,xmax+1e-9,tick):
        ends=pixels([[x,-limit],[x,limit]]).astype(int);cv2.line(out,tuple(ends[0]),tuple(ends[1]),(60,65,70),1)
        cv2.putText(out,f'{x:g}',(7,int(ends[0,1])+4),0,.32,(195,205,220),1)
    for y in np.arange(np.ceil(-limit/tick)*tick,limit+1e-9,tick):
        ends=pixels([[xmin,y],[xmax,y]]).astype(int);cv2.line(out,tuple(ends[0]),tuple(ends[1]),(60,65,70),1)
        cv2.putText(out,f'{y:g}',(int(ends[0,0])-10,min(h-4,round(bottom)+14)),0,.32,(195,205,220),1)
    for boundary,color in ((plan.left,LEFT),(plan.right,RIGHT)):
        if boundary is None:continue
        points=boundary.points_cm[::max(1,len(boundary.points_cm)//3000)]
        xy=pixels(points).astype(int)
        good=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        out[xy[good,1],xy[good,0]]=color if stage==0 else (170,180,190)
    for points in getattr(plan,'partial_points_cm',{}).values():
        xy=pixels(points).astype(int)
        good=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        out[xy[good,1],xy[good,0]]=(180,120,255)
    if stage>=1:
        for side,points,color,inferred in boundary_curves(plan):polyline(out,pixels(points),color,2,False)
    if stage>=2 and plan.usable:polyline(out,pixels(plan.path_cm),PATH,2,False)
    origin=tuple(np.rint(pixels([[0,0]])[0]).astype(int));cv2.circle(out,origin,4,(220,220,220),-1)
    cv2.putText(out,'X forward / Y left (cm); fixed axes',(8,17),0,.37,(235,240,245),1,cv2.LINE_AA)
    return out


def dashboard(p,lanes,plan,mapping,inverse,index,total,fps,ms):
    source=green_overlay(p.original,plan,mapping,inverse);bev=green_overlay(p.bev,plan,mapping)
    status=np.full((360,480,3),(24,28,35),np.uint8)
    lines=[f'Frame {index+1}/{total} | {index/fps:.2f}s',f'Path: {plan.status}',
        'Joint B-spline center / width' if getattr(plan,'joint_fit_accepted',False) else 'Current boundary model',
        'Gray dots: actual observed paint','Pink: short current observation','Green: continuous center path',
        f'Path adjustment: {getattr(plan,"max_path_adjustment_cm",0):.2f} cm',
        f'Pipeline: {1000/max(ms,.001):.1f} FPS','X: forward / Y: left (cm)','Origin: BEV bottom reference']
    for j,line in enumerate(lines):cv2.putText(status,line,(12,28+j*30),0,.45,(230,239,233),1,cv2.LINE_AA)
    top=np.hstack((tile_keep(source,'Original / green target path'),tile_keep(bev,'BEV / same scale, wider canvas'),tile_keep(lanes.debug,'Sliding windows / current pixels')))
    bottom=np.hstack((metric_with_partials(plan,0),metric_with_partials(plan,2),status))
    footer=np.full((80,1440,3),(22,29,24),np.uint8)
    cv2.putText(footer,'Week 3 | observed pixels -> cm lane functions -> continuous target path',(12,31),0,.65,(225,239,229),1,cv2.LINE_AA)
    cv2.putText(footer,'Recorded-video planning result; inferred and partial observations remain identified.',(12,61),0,.53,(205,224,211),1,cv2.LINE_AA)
    return np.vstack((top,bottom,footer))


def save_final_plots(folder,index,plan,stages=(0,1,2)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for stage,name,title in [(0,'02_cm_points','Observed lane pixels in centimeters'),(1,'03_metric_functions','Quadratic functions from observed points'),(2,'04_planned_path','Green target path')]:
        if stage not in stages:continue
        fig,ax=plt.subplots(figsize=(6.6,6.4))
        fig.subplots_adjust(left=.15,right=.97,bottom=.12,top=.93)
        for side,boundary,color in [('left',plan.left,'#197fab'),('right',plan.right,'#c38c13')]:
            if boundary is None:continue
            pp=boundary.points_cm[::max(1,len(boundary.points_cm)//2000)]
            ax.scatter(pp[:,1],pp[:,0],s=2,alpha=.45 if stage==2 else .30,
                color='#7c858b' if stage==2 else color,label=f'{side} observed points')
        for side,points in getattr(plan,'partial_points_cm',{}).items():
            pp=points[::max(1,len(points)//600)];ax.scatter(pp[:,1],pp[:,0],s=4,color='#b54da5',alpha=.5,label=f'{side} partial points')
        if stage==1:
            for side,boundary,color in [('left',plan.left,'#197fab'),('right',plan.right,'#c38c13')]:
                if boundary is None:continue
                xx=np.linspace(*boundary.interval_cm,180)
                ax.plot(boundary.y(xx),xx,color=color,lw=2,label=f'{side} observed quadratic')
        elif stage>=2:
            for side,points,color,inferred in boundary_curves(plan):
                label=f'{side} completed model (inferred)' if inferred else f'{side} boundary model'
                ax.plot(points[:,1],points[:,0],'-',color='#197fab' if side=='left' else '#c38c13',lw=2,label=label)
        if stage>=2 and plan.usable:
            if getattr(plan,'stabilization_applied',False):
                pp=plan.raw_path_cm;ax.plot(pp[:,1],pp[:,0],color='#929b94',lw=1,alpha=.7,label='current unfiltered path')
            pp=plan.path_cm;ax.plot(pp[:,1],pp[:,0],'-',color='#159148',lw=2.7,label='continuous center path')
        ax.scatter([0],[0],s=22,color='#334037',label='BEV reference')
        bounds=metric_view_bounds(plan)
        ax.set(xlim=(bounds['y_limit'],-bounds['y_limit']),ylim=(bounds['x_min'],bounds['x_max']),xlabel='Lateral Y (cm, +left)',ylabel='Forward X (cm)',title=f'Frame {index}: {title}')
        tick=bounds['tick_cm']
        ax.set_xticks(np.arange(np.ceil(-bounds['y_limit']/tick)*tick,bounds['y_limit']+1e-9,tick))
        ax.set_yticks(np.arange(np.ceil(bounds['x_min']/tick)*tick,bounds['x_max']+1e-9,tick))
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.legend(fontsize=9,loc='upper right')
        ax.xaxis.label.set_size(13);ax.yaxis.label.set_size(13);ax.tick_params(labelsize=11)
        fig.savefig(folder/f'{name}.png',dpi=170);plt.close(fig)
