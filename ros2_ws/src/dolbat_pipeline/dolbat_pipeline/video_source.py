"""Original recording publisher with bounded end-to-end frame credits."""
from pathlib import Path
import hashlib,math,time
import cv2
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from std_msgs.msg import Header,UInt32
from cv_bridge import CvBridge
from dolbat_interfaces.msg import VideoFrame
from .common import frame_qos,ack_qos,spin_node

ORIGINAL_SHA256='c99bd02d43f12dbcd093b808ef7bdc6c85d17c3b5620808213c5e2e2e14cc66f'


class VideoSourceNode(Node):
    def __init__(self):
        super().__init__('video_source')
        self.declare_parameter('video','data/dolbat_toyproject.mp4')
        self.declare_parameter('replay_rate',1.0)
        self.declare_parameter('max_in_flight',4)
        self.declare_parameter('stall_timeout_sec',120.0)
        path=Path(self.get_parameter('video').value).expanduser().resolve()
        if not path.is_file():raise FileNotFoundError(f'Original video missing: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ORIGINAL_SHA256:
            raise ValueError('This calibrated launch requires the original dolbat_toyproject.mp4')
        self.capture=cv2.VideoCapture(str(path))
        if not self.capture.isOpened():raise ValueError('Cannot open original video')
        self.fps=float(self.capture.get(cv2.CAP_PROP_FPS));self.total=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.total!=564 or abs(self.fps-24)>1e-9 or (width,height)!=(640,480):
            raise ValueError('Original 640x480,24FPS,564-frame recording required')
        self.rate=float(self.get_parameter('replay_rate').value)
        self.window=int(self.get_parameter('max_in_flight').value)
        self.timeout=float(self.get_parameter('stall_timeout_sec').value)
        if not math.isfinite(self.rate) or self.rate<0 or not 1<=self.window<=4 or not math.isfinite(self.timeout) or self.timeout<=0:
            raise ValueError('replay_rate>=0, max_in_flight 1..4 and positive timeout required')
        self.publisher=self.create_publisher(VideoFrame,'/dolbat/video',frame_qos())
        self.ack_sub=self.create_subscription(UInt32,'/dolbat/frame_done',self.acknowledge,ack_qos())
        self.bridge=CvBridge();self.next_frame=0;self.next_ack=0;self.pending=set();self.eos=False
        self.last_progress=time.monotonic();self.start_ns=None
        period=max(.001,1/(self.fps*self.rate)) if self.rate else .001
        self.timer=self.create_timer(period,self.tick)
        self.get_logger().info(f'Ready: {self.total} frames, replay_rate={self.rate:g}, max_in_flight={self.window}')

    def header(self,index):
        stamp_ns=self.start_ns+round(index/self.fps*1_000_000_000)
        header=Header();header.frame_id='dolbat_camera'
        header.stamp=Time(sec=stamp_ns//1_000_000_000,nanosec=stamp_ns%1_000_000_000)
        return header

    def acknowledge(self,message):
        index=int(message.data)
        if index<self.next_ack:return
        if index!=self.next_ack or index not in self.pending:
            raise RuntimeError(f'Unexpected frame acknowledgement {index}; expected {self.next_ack}')
        self.pending.remove(index);self.next_ack+=1;self.last_progress=time.monotonic()

    def tick(self):
        if self.eos:return
        if time.monotonic()-self.last_progress>self.timeout:
            raise TimeoutError('No frame progress: check downstream nodes and ROS discovery')
        # The image has two consumers. Transient-local histories in subsequent
        # links retain at least the four permitted in-flight frames at discovery.
        if self.publisher.get_subscription_count()<2 or self.count_publishers('/dolbat/frame_done')<1:return
        if self.start_ns is None:self.start_ns=self.get_clock().now().nanoseconds
        if self.next_frame==self.total:
            if self.pending:return
            message=VideoFrame();message.header=self.header(self.total)
            message.frame_index=self.total;message.source_fps=self.fps;message.end_of_stream=True
            self.publisher.publish(message);self.eos=True;self.timer.cancel();self.capture.release()
            self.get_logger().info(f'End-of-stream after all {self.next_ack} acknowledgements; no frames dropped')
            return
        if len(self.pending)>=self.window:return
        ok,frame=self.capture.read()
        if not ok:raise RuntimeError(f'Video ended before advertised frame {self.next_frame}')
        message=VideoFrame();message.header=self.header(self.next_frame)
        message.frame_index=self.next_frame;message.source_fps=self.fps
        message.image=self.bridge.cv2_to_imgmsg(frame,encoding='bgr8');message.image.header=message.header
        self.pending.add(self.next_frame);self.publisher.publish(message)
        self.next_frame+=1

    def destroy_node(self):
        self.capture.release()
        return super().destroy_node()


def main(args=None):spin_node(VideoSourceNode,args)
