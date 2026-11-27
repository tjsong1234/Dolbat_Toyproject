"""Shared lossless frame QoS and ordered-stage guards."""
import math
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile,HistoryPolicy,ReliabilityPolicy,DurabilityPolicy


def frame_qos():
    # Source credit is limited to four frames, so a depth of eight retains all
    # in-flight frames while downstream nodes discover each other at startup.
    return QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=8,
                      reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def ack_qos():
    return frame_qos()


def copy_frame_fields(source,destination):
    destination.header=source.header
    destination.frame_index=int(source.frame_index)
    destination.source_fps=float(source.source_fps)
    destination.end_of_stream=bool(source.end_of_stream)
    return destination


class OrderedStage(Node):
    def __init__(self,name):
        super().__init__(name)
        self.expected_index=0
        self.source_fps=None
        self.ended=False

    def accept(self,message):
        index=int(message.frame_index);fps=float(message.source_fps)
        if self.ended:raise RuntimeError(f'{self.get_name()}: data received after end-of-stream')
        if index!=self.expected_index:
            raise RuntimeError(f'{self.get_name()}: expected frame {self.expected_index}, received {index}')
        if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid source FPS')
        if self.source_fps is None:self.source_fps=fps
        elif abs(self.source_fps-fps)>1e-9:raise ValueError('Source FPS changed within a recording')
        if message.end_of_stream:self.ended=True
        else:self.expected_index+=1
        return not bool(message.end_of_stream)


def spin_node(node_type,args=None):
    rclpy.init(args=args);node=None
    try:
        node=node_type();rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):
        pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
