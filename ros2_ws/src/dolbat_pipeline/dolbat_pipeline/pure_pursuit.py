"""Week 4: rear-axle LP and offline steering; no actuator command topic."""
from copy import deepcopy
import cv2
from geometry_msgs.msg import PointStamped
from dolbat_interfaces.msg import PathFrame,ControlFrame
from dolbat_core.runtime import Runtime
from dolbat_core.week4_control import bev_cm_to_rear_m,calculate_pure_pursuit
from .common import OrderedStage,frame_qos,copy_frame_fields,spin_node
from .codecs import points,encode_control


class PurePursuitNode(OrderedStage):
    def __init__(self):
        super().__init__('pure_pursuit')
        cv2.setNumThreads(2);self.runtime=Runtime()
        self.publisher=self.create_publisher(ControlFrame,'/dolbat/control',frame_qos())
        self.target_publisher=self.create_publisher(PointStamped,'/dolbat/lookahead_point',frame_qos())
        self.subscription=self.create_subscription(PathFrame,'/dolbat/path',self.receive,frame_qos())

    def receive(self,message):
        active=self.accept(message)
        if active:
            rear=bev_cm_to_rear_m(points(message.path_bev_cm),*self.runtime.translation)
            result=calculate_pure_pursuit(rear,self.runtime.control_config)
            output=encode_control(result,message)
            if result.valid:
                target=PointStamped();target.header=deepcopy(message.header);target.header.frame_id='rear_axle'
                target.point.x=float(result.target_m[0]);target.point.y=float(result.target_m[1])
                self.target_publisher.publish(target)
                self.get_logger().info(f'frame={message.frame_index} LP=({result.target_m[0]:.5f},{result.target_m[1]:+.5f})m '
                                       f'steering={result.steering_deg:+.5f}deg ({result.steering_rad:+.6f}rad)')
            else:self.get_logger().warning(f'frame={message.frame_index} steering=UNAVAILABLE {result.status}')
        else:output=copy_frame_fields(message,ControlFrame())
        self.publisher.publish(output)


def main(args=None):spin_node(PurePursuitNode,args)
