"""ROS-independent construction of the verified four-stage CPU pipeline."""
from pathlib import Path
import json
import cv2
import numpy as np
import yaml
from .road_surface_preprocessing import RoadSurfacePreprocessor
from .final_lane_detection import FinalLaneDetector
from .joint_bspline_planning import JointBSplineLanePlanner
from .metric_mapping import MetricMapping
from .final_visualization import attach_partials, set_metric_axes
from .week4_geometry import rear_reference
from .week4_control import PurePursuitConfig, bev_cm_to_rear_m, calculate_pure_pursuit


class Runtime:
    def __init__(self, config_dir=None, fps=24., image_size=(640,480)):
        resources=Path(__file__).resolve().parent/'resources'
        directory=Path(config_dir) if config_dir else resources/'config'
        read=lambda name:yaml.safe_load((directory/name).read_text(encoding='utf-8'))
        self.base=read('settings.yaml');self.geo=read('geometry_final.yaml')
        self.profile=read('perception_improved.yaml');self.week=read('week3_joint.yaml')
        self.control_settings=read('week4.yaml')
        self.calibration=json.loads((resources/'week3_reference_calibration.json').read_text(encoding='utf-8'))
        self.calibration['origin_u_px']+=self.week['geometry_translation_u_px']
        self.calibration['canvas_size_px']=self.geo['preprocessing']['bev_size']
        self.calibration['origin_note']='Translated BEV reference; rear axle derived from original image bottom.'
        self.mapping=MetricMapping(**{key:self.calibration[key] for key in
            ('lateral_cm_per_px','forward_cm_per_px','origin_u_px','origin_v_px','status')})
        self.pre=RoadSurfacePreprocessor(self.base['preprocessing'],self.geo,self.profile['marking_suppression'])
        self.detector=FinalLaneDetector(self.geo['lane_detection'],self.geo['lane_validation'],self.week['lane_context'])
        self.planner=JointBSplineLanePlanner(self.mapping,self.week['planning'],
            self.calibration['centerline_lane_width_cm'],self.geo['lane_detection']['windows'],
            self.geo['preprocessing']['bev_size'][1],fps=float(fps))
        self.control_config=PurePursuitConfig(self.control_settings['lookahead_m'],self.control_settings['wheelbase_m'])
        width,height=map(int,image_size)
        if (width,height)!=(640,480):
            raise ValueError('The supplied calibration requires the original 640x480 recording.')
        anchors=np.float32(self.pre.cfg['source_points'])*[width-1,height-1]
        size=tuple(self.pre.cfg['bev_size'])
        destination=np.float32(self.pre.cfg['destination_points'])*[size[0]-1,size[1]-1]
        self.matrix=cv2.getPerspectiveTransform(anchors.astype(np.float32),destination.astype(np.float32))
        self.inverse=np.linalg.inv(self.matrix)
        self.geometry=rear_reference(self.matrix,self.mapping,(width,height),self.control_settings['rear_to_image_bottom_m'])
        self.translation=np.array(self.geometry['rear_translation_m'])

    def plan(self, lanes):
        result=self.planner.plan(lanes)
        attach_partials(result,lanes,self.mapping)
        set_metric_axes(result,self.week.get('visualization',{}).get('metric_axes'))
        return result

    def rear_path(self, path_cm):
        return bev_cm_to_rear_m(path_cm,*self.translation)

    def control(self, path_cm):
        return calculate_pure_pursuit(self.rear_path(path_cm),self.control_config)
