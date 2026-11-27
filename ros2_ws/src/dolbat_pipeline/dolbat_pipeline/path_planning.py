"""Week 3: cm mapping and joint B-spline centre/normal-width fitting."""
from copy import deepcopy
import math
import cv2
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from dolbat_interfaces.msg import LaneFrame,PathFrame
from dolbat_core.runtime import Runtime
from dolbat_core.week4_control import bev_cm_to_rear_m
from .common import OrderedStage,frame_qos,copy_frame_fields,spin_node
from .codecs import decode_lanes,encode_plan


class PathPlanningNode(OrderedStage):
    def __init__(self):
        super().__init__('path_planning')
        cv2.setNumThreads(2);self.runtime=Runtime()
        self.publisher=self.create_publisher(PathFrame,'/dolbat/path',frame_qos())
        self.path_publisher=self.create_publisher(Path,'/dolbat/path_rear',frame_qos())
        self.subscription=self.create_subscription(LaneFrame,'/dolbat/lanes',self.receive,frame_qos())

    def receive(self,message):
        active=self.accept(message)
        if active:
            lanes=decode_lanes(message);plan=self.runtime.plan(lanes)
            output=encode_plan(plan,message)
            standard=Path();standard.header=deepcopy(message.header);standard.header.frame_id='rear_axle'
            rear=bev_cm_to_rear_m(plan.path_cm,*self.runtime.translation)
            if len(rear)>=2:
                direction=np.gradient(rear,axis=0)
                yaw=np.arctan2(direction[:,1],direction[:,0])
                for (x,y),angle in zip(rear,yaw):
                    pose=PoseStamped();pose.header=standard.header
                    pose.pose.position.x=float(x);pose.pose.position.y=float(y)
                    pose.pose.orientation.z=math.sin(float(angle)/2);pose.pose.orientation.w=math.cos(float(angle)/2)
                    standard.poses.append(pose)
            self.path_publisher.publish(standard)
        else:output=copy_frame_fields(message,PathFrame())
        self.publisher.publish(output)


def main(args=None):spin_node(PathPlanningNode,args)
