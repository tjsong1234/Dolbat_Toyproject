"""Synchronize four indexed streams, record every frame, then acknowledge it."""
from collections import Counter,deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import csv,json,subprocess,time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32
from cv_bridge import CvBridge
from dolbat_interfaces.msg import VideoFrame,PathFrame,ControlFrame
from dolbat_core.runtime import Runtime
from dolbat_core.video_output import AsyncVideoSink
from dolbat_core.week4_visualization import compose,plot_transform
from dolbat_core.week4_control import bev_cm_to_rear_m
from dolbat_core.sliding_visualization import save_png
from .common import frame_qos,ack_qos
from .codecs import decode_plan,decode_control
from .joining import FrameJoiner


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization')
        self.declare_parameter('output','outputs/ros2_run')
        self.declare_parameter('display',False)
        self.output=Path(self.get_parameter('output').value).expanduser().resolve()
        if self.output.exists() and any(self.output.iterdir()):raise ValueError('Output must be empty or new')
        for sub in ('records','images','graphs'):(self.output/sub).mkdir(parents=True,exist_ok=True)
        self.display=bool(self.get_parameter('display').value)
        self.runtime=Runtime();self.bridge=CvBridge();self.joiner=FrameJoiner()
        self.ack=self.create_publisher(UInt32,'/dolbat/frame_done',ack_qos())
        self.subscriptions_=[]
        for kind,topic,cls in [('video','/dolbat/video',VideoFrame),('debug','/dolbat/sliding_debug',VideoFrame),
                               ('path','/dolbat/path',PathFrame),('control','/dolbat/control',ControlFrame)]:
            self.subscriptions_.append(self.create_subscription(cls,topic,lambda msg,k=kind:self.receive(k,msg),frame_qos()))
        self.started=None;self.finished=False;self.failed=False;self.sink=None
        self.rows=[];self.paths=[];self.axes=[];self.history=[];self.completions=[]
        self.perception=Counter();self.planning=Counter();self.controls=Counter()
        self.fps=None;self.examples=set(self.runtime.control_settings['example_frames'])

    def receive(self,kind,msg):
        try:
            if self.finished:return
            if kind=='video' and self.started is None and not msg.end_of_stream:self.started=time.perf_counter()
            if msg.end_of_stream:
                if kind=='control':self.finalize(int(msg.frame_index))
                return
            stamp=(msg.header.frame_id,msg.header.stamp.sec,msg.header.stamp.nanosec)
            for index,group in self.joiner.add(kind,msg.frame_index,stamp,msg):self.render(index,group)
        except Exception as exc:
            self.failed=True;self.get_logger().error(str(exc))
            self.output.joinpath('failure.txt').write_text(repr(exc),encoding='utf-8')
            # Leave the callback before shutting down the global executor.

    def render(self,index,group):
        if self.fps is None:
            self.fps=float(group['video'].source_fps)
            self.sink=AsyncVideoSink([(self.output/'01_final_four_panels.raw.mp4',(1280,1024)),
                                      (self.output/'02_green_path_control.raw.mp4',(1280,544))],self.fps)
        if any(abs(float(m.source_fps)-self.fps)>1e-9 for m in group.values()):raise ValueError('Source FPS mismatch')
        frame=self.bridge.imgmsg_to_cv2(group['video'].image,desired_encoding='bgr8')
        debug=self.bridge.imgmsg_to_cv2(group['debug'].image,desired_encoding='bgr8')
        plan=decode_plan(group['path']);result=decode_control(group['control'])
        rear=bev_cm_to_rear_m(plan.path_cm,*self.runtime.translation)
        now=time.perf_counter();self.completions.append(now)
        recent=self.completions[-24:]
        ms=(recent[-1]-recent[0])*1000/(len(recent)-1) if len(recent)>1 else (now-self.started)*1000
        self.history.append(result.steering_deg)
        lanes=SimpleNamespace(debug=debug,status=group['path'].perception_status)
        final,green,control,metric=compose(frame,None,lanes,plan,self.runtime.mapping,self.runtime.inverse,
            rear,result,self.runtime.translation,self.runtime.control_settings,index,564,self.fps,max(ms,.001),self.history)
        snapshots=[]
        if index in self.examples:
            snapshots=[(self.output/'images'/f'{name}_{index:04d}.png',im)
                       for name,im in [('final',final),('green',green),('control',control),('metric',metric)]]
        if index==100:snapshots.append((self.output/'images/geometry.png',metric))
        self.sink.write(final,green,pngs=snapshots)
        if self.display:
            cv2.imshow('Dolbat ROS2 / original + BEV + path + steering',final)
            if cv2.waitKey(1)&255==27:raise RuntimeError('Stopped by user before completing the video')
        row={'frame':index,'time_s':index/self.fps,**asdict(result),
             'target_x_m':result.target_m[0] if result.valid else None,
             'target_y_m':result.target_m[1] if result.valid else None,
             'perception_status':lanes.status,'planning_status':plan.status}
        row.pop('target_m');self.rows.append(row);self.paths.append(rear)
        self.axes.append({'frame':index,**plot_transform(self.runtime.control_settings['visualization'])})
        self.perception[lanes.status]+=1;self.planning[plan.status]+=1;self.controls[result.status]+=1
        self.ack.publish(UInt32(data=index))
        if (index+1)%100==0:self.get_logger().info(f'Recorded {index+1}/564; frame order preserved')

    def finalize(self,total):
        self.joiner.finish(total)
        if total!=564 or len(self.rows)!=total:raise ValueError('Original full 564-frame recording required')
        if self.sink is None:raise ValueError('No recorded frames')
        self.sink.close();elapsed=time.perf_counter()-self.started
        self.finished=True
        self.write_csv('control.csv',self.rows);self.write_csv('plot_axes.csv',self.axes)
        with (self.output/'records/rear_paths.csv').open('w',encoding='utf-8',newline='') as file:
            writer=csv.writer(file);writer.writerow(['frame','time_s','sample','forward_X_m','left_Y_m'])
            writer.writerows((i,i/self.fps,j,float(x),float(y)) for i,p in enumerate(self.paths) for j,(x,y) in enumerate(p))
        videos={}
        for stem in ('01_final_four_panels','02_green_path_control'):
            path=self.output/(stem+'.mp4')
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2','-i',str(self.output/(stem+'.raw.mp4')),
                            '-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-threads','2',str(path)],check=True)
            cap=cv2.VideoCapture(str(path));count=0;actual_fps=cap.get(cv2.CAP_PROP_FPS)
            size=[int(cap.get(3)),int(cap.get(4))]
            while True:
                ok,im=cap.read()
                if not ok:break
                if float(im.std())<2:raise ValueError('Blank output frame')
                count+=1
            cap.release()
            if count!=total or abs(actual_fps-self.fps)>1e-6:raise ValueError('Output video incomplete')
            videos[path.name]={'decoded_frames':count,'fps':actual_fps,'size':size}
        self.plot()
        angles=np.array([r['steering_deg'] for r in self.rows if r['valid']],float)
        intervals=np.diff(self.completions)*1000
        summary={'frames':total,'fps':self.fps,'duration_s':total/self.fps,'transport':'ROS2 Humble, six separate nodes',
            'pipeline_fps':total/elapsed,'export_fps':total/elapsed,'export_seconds':elapsed,
            'pipeline_fps_scope':'First original ROS frame arrival through last output worker drain; excludes final H264 conversion and final CSV/graph export.',
            'control_counts':dict(self.controls),'perception_counts':dict(self.perception),'planning_counts':dict(self.planning),
            'control_config':{k:self.runtime.control_settings[k] for k in ('rear_to_image_bottom_m','lookahead_m','wheelbase_m')},
            'geometry':self.runtime.geometry,'writer':self.sink.stats(),'frames_dropped':0,
            'production_interval_p95_ms':float(np.percentile(intervals,95)),
            'production_interval_max_ms':float(intervals.max()),'production_intervals_over_24fps_budget':int(np.count_nonzero(intervals>1000/self.fps)),
            'steering_min_deg':float(angles.min()) if len(angles) else None,'steering_max_deg':float(angles.max()) if len(angles) else None,
            'videos':videos,'scope':'Recorded video only, no vehicle command topic or actuation.'}
        (self.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        self.get_logger().info(f'COMPLETE {total} frames, {total/elapsed:.2f} FPS; videos fully decoded')

    def write_csv(self,name,rows):
        with (self.output/'records'/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4))
        ax.plot([r['time_s'] for r in self.rows],[np.nan if r['steering_deg'] is None else r['steering_deg'] for r in self.rows],color='#217349')
        ax.set(xlabel='Recording time (s)',ylabel='Steering angle (deg)',xlim=(0,564/self.fps),ylim=(-25,25),
               title='ROS2 Pure Pursuit | positive left, negative right')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(self.output/'graphs/steering.png',dpi=160);plt.close(fig)

    def close(self):
        if self.sink is not None:self.sink.close()
        if self.display:cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args);node=None
    try:
        node=VisualizationNode()
        while rclpy.ok() and not node.finished and not node.failed:
            rclpy.spin_once(node,timeout_sec=.1)
        if node.failed:raise RuntimeError('Visualization/recording failed')
    except KeyboardInterrupt:pass
    finally:
        if node is not None:node.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
