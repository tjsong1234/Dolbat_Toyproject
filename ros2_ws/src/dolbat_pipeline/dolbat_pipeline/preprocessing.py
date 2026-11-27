"""Week 1: ROI, planar BEV and paint-noise suppression."""
import cv2
from cv_bridge import CvBridge
from dolbat_interfaces.msg import VideoFrame,PreprocessedFrame
from dolbat_core.runtime import Runtime
from .common import OrderedStage,frame_qos,copy_frame_fields,spin_node


class PreprocessingNode(OrderedStage):
    def __init__(self):
        super().__init__('preprocessing')
        cv2.setNumThreads(2);self.runtime=Runtime();self.bridge=CvBridge()
        self.publisher=self.create_publisher(PreprocessedFrame,'/dolbat/preprocessed',frame_qos())
        self.subscription=self.create_subscription(VideoFrame,'/dolbat/video',self.receive,frame_qos())

    def receive(self,message):
        active=self.accept(message);output=copy_frame_fields(message,PreprocessedFrame())
        if active:
            frame=self.bridge.imgmsg_to_cv2(message.image,desired_encoding='bgr8')
            if frame.shape!=(480,640,3):raise ValueError('Unexpected source dimensions')
            processed=self.runtime.pre.process(frame)
            output.source_width=frame.shape[1];output.source_height=frame.shape[0]
            for name,array in [('binary',processed.binary),('paint',self.runtime.pre.paint),
                               ('crosswalk',self.runtime.pre.suppression.crosswalk)]:
                image=self.bridge.cv2_to_imgmsg(array,encoding='mono8');image.header=message.header
                setattr(output,name,image)
        self.publisher.publish(output)


def main(args=None):spin_node(PreprocessingNode,args)
