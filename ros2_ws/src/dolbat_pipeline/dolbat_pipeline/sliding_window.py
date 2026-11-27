"""Week 2: ordered Sliding Window and observed/partial lane fitting."""
import cv2
from cv_bridge import CvBridge
from dolbat_interfaces.msg import PreprocessedFrame,LaneFrame,VideoFrame
from dolbat_core.runtime import Runtime
from .common import OrderedStage,frame_qos,copy_frame_fields,spin_node
from .codecs import encode_lanes


class SlidingWindowNode(OrderedStage):
    def __init__(self):
        super().__init__('sliding_window')
        cv2.setNumThreads(2);self.runtime=Runtime();self.bridge=CvBridge()
        self.publisher=self.create_publisher(LaneFrame,'/dolbat/lanes',frame_qos())
        self.debug_publisher=self.create_publisher(VideoFrame,'/dolbat/sliding_debug',frame_qos())
        self.subscription=self.create_subscription(PreprocessedFrame,'/dolbat/preprocessed',self.receive,frame_qos())

    def receive(self,message):
        active=self.accept(message)
        debug=copy_frame_fields(message,VideoFrame())
        if active:
            binary=self.bridge.imgmsg_to_cv2(message.binary,desired_encoding='mono8')
            paint=self.bridge.imgmsg_to_cv2(message.paint,desired_encoding='mono8')
            crosswalk=self.bridge.imgmsg_to_cv2(message.crosswalk,desired_encoding='mono8')
            if binary.shape!=paint.shape or binary.shape!=crosswalk.shape:raise ValueError('Perception masks have different dimensions')
            lanes=self.runtime.detector.detect(binary,paint,crosswalk)
            output=encode_lanes(lanes,message)
            debug.image=self.bridge.cv2_to_imgmsg(lanes.debug,encoding='bgr8');debug.image.header=message.header
        else:output=copy_frame_fields(message,LaneFrame())
        self.publisher.publish(output);self.debug_publisher.publish(debug)


def main(args=None):spin_node(SlidingWindowNode,args)
