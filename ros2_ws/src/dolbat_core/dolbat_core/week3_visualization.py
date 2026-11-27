"""Live CPU panels and report plots for cm lane functions and a planned path."""
from pathlib import Path
import cv2
import numpy as np
from .sliding_visualization import save_png

LEFT=(255,170,30);RIGHT=(20,225,255);PATH=(65,220,90)


def tile_keep(image,title,size=(480,360)):
    if image.ndim==2:image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    w,h=size;scale=min(w/image.shape[1],(h-26)/image.shape[0])
    resized=cv2.resize(image,None,fx=scale,fy=scale)
    out=np.full((h,w,3),(22,26,33),np.uint8)
    x=(w-resized.shape[1])//2;y=26+(h-26-resized.shape[0])//2
    out[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    cv2.putText(out,title,(6,18),0,.45,(235,240,245),1,cv2.LINE_AA)
    return out


def polyline(image,points,color,thickness=2,dashed=False):
    points=np.asarray(points)
    if not len(points):return
    good=np.isfinite(points).all(axis=1)&(abs(points).max(axis=1)<100000)
    points=points[good].astype(np.int32)
    if dashed:
        for i in range(0,len(points)-2,8):cv2.polylines(image,[points[i:i+5]],False,color,thickness,cv2.LINE_AA)
    elif len(points)>1:cv2.polylines(image,[points],False,color,thickness,cv2.LINE_AA)


def draw_metric_overlay(image,plan,mapping,inverse=None):
    out=image.copy()
    def project(xy):
        uv=mapping.cm_to_pixels(xy).astype(np.float32)
        return cv2.perspectiveTransform(uv.reshape(-1,1,2),inverse).reshape(-1,2) if inverse is not None else uv
    for boundary,color in ((plan.left,LEFT),(plan.right,RIGHT)):
        if boundary is None:continue
        xx=np.linspace(*boundary.interval_cm,180)
        polyline(out,project(np.column_stack((xx,boundary.y(xx)))),color,2)
    if plan.inferred_boundary_cm is not None:
        polyline(out,project(plan.inferred_boundary_cm),LEFT if plan.left is None else RIGHT,2,True)
    if plan.usable:polyline(out,project(plan.path_cm),PATH,3,plan.status!='OBSERVED_PAIR')
    return out


def metric_panel(plan,stage=2,size=(480,360)):
    w,h=size;out=np.full((h,w,3),(24,28,35),np.uint8)
    scale=min((w-75)/300,(h-65)/320);cx=w//2;bottom=h-30
    def pixels(xy):return np.column_stack((cx-np.asarray(xy)[:,1]*scale,bottom-np.asarray(xy)[:,0]*scale))
    for x in range(0,321,50):
        p=pixels([[x,-150],[x,150]]).astype(int);cv2.line(out,tuple(p[0]),tuple(p[1]),(60,65,70),1)
        cv2.putText(out,str(x),(12,int(p[0,1])+4),0,.35,(195,205,220),1)
    for y in range(-100,101,50):
        p=pixels([[0,y],[320,y]]).astype(int);cv2.line(out,tuple(p[0]),tuple(p[1]),(60,65,70),1)
        cv2.putText(out,str(y),(int(p[0,0])-9,h-12),0,.35,(195,205,220),1)
    for boundary,color in ((plan.left,LEFT),(plan.right,RIGHT)):
        if boundary is None:continue
        points=boundary.points_cm
        xy=pixels(points[::max(1,len(points)//3000)]).astype(int)
        good=(xy[:,0]>=45)&(xy[:,0]<w-10)&(xy[:,1]>=25)&(xy[:,1]<h-27)
        out[xy[good,1],xy[good,0]]=color
        if stage>=1:
            xx=np.linspace(*boundary.interval_cm,160);polyline(out,pixels(np.column_stack((xx,boundary.y(xx)))),color,2)
    if stage>=2:
        if plan.inferred_boundary_cm is not None:polyline(out,pixels(plan.inferred_boundary_cm),(145,150,160),1,True)
        if plan.usable:polyline(out,pixels(plan.path_cm),PATH,2,plan.status!='OBSERVED_PAIR')
    cv2.circle(out,(cx,bottom),4,(220,220,220),-1)
    cv2.putText(out,'X forward / Y left (cm); reference origin',(8,17),0,.39,(235,240,245),1,cv2.LINE_AA)
    return out


def save_plots(folder,frame,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for stage,name,title in [(0,'02_cm_points','Observed sliding-window pixels in cm'),
                             (1,'03_metric_functions','Quadratic lane functions fitted AFTER cm conversion'),
                             (2,'04_planned_path','Continuous path from metric lane functions')]:
        fig,ax=plt.subplots(figsize=(6.4,6.4),layout='constrained')
        for side,boundary,color in (('left',plan.left,'#1689c4'),('right',plan.right,'#c48a00')):
            if boundary is None:continue
            pp=boundary.points_cm[::max(1,len(boundary.points_cm)//2500)]
            ax.scatter(pp[:,1],pp[:,0],s=2,alpha=.25,color=color,label=f'{side} observed pixels')
            if stage>=1:
                xx=np.linspace(*boundary.interval_cm,200);ax.plot(boundary.y(xx),xx,color=color,lw=2,label=f'{side} quadratic')
        if stage>=2:
            if plan.inferred_boundary_cm is not None:
                pp=plan.inferred_boundary_cm;ax.plot(pp[:,1],pp[:,0],'--',color='#999fa5',lw=1.3,label='unobserved partner estimate')
            if plan.usable:
                pp=plan.path_cm;ax.plot(pp[:,1],pp[:,0],'-' if plan.status=='OBSERVED_PAIR' else '--',color='#1a9c55',lw=2.5,label='planned path' if plan.status=='OBSERVED_PAIR' else 'estimated planned path')
        ax.scatter([0],[0],color='black',s=25,label='BEV reference origin')
        ax.set(xlim=(150,-150),ylim=(0,320),xlabel='Lateral Y (cm, +left)',ylabel='Forward X (cm)',title=f'Frame {frame}: {title}')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.25);ax.legend(fontsize=7,loc='upper right')
        fig.savefig(folder/f'{name}.png',dpi=160);plt.close(fig)


def status_panel(plan,mapping,perception,frame):
    out=np.full((360,480,3),(24,28,35),np.uint8)
    lines=[f'Frame index: {frame}',f'Perception: {perception}',f'Planning: {plan.status}',
        f'Lateral: {mapping.lateral_cm_per_px:.6f} cm / pixel',f'Forward: {mapping.forward_cm_per_px:.6f} cm / pixel',
        'Curve: Y(cm) = A X(cm)^2 + B X(cm) + C',
        'Blue=left / yellow=right / green=path',
        'Green dashed = path from ONE observed side',
        'Origin: BEV bottom center reference',
        'Rear axle location is not measured.',
        'Course-reference scale; no steering command.']
    if plan.usable:lines.insert(5,f'Path X range: {plan.path_cm[0,0]:.1f} .. {plan.path_cm[-1,0]:.1f} cm')
    for i,line in enumerate(lines):cv2.putText(out,line,(12,25+i*27),0,.42,(230,235,240),1,cv2.LINE_AA)
    return out
