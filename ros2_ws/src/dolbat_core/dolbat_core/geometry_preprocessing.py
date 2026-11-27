"""A larger processing ROI with independent, padded planar BEV anchors."""
from copy import deepcopy
import cv2
import numpy as np
from .preprocessing import Preprocessed
from .perception_review import candidate_mask
from .marking_suppression import suppress_markings


def retain_road_adjacent_components(candidate,nearby_road,min_area=45,fraction=.12):
    """Gate a complete paint component by road contact; do not hollow its center."""
    n,labels,stats,_=cv2.connectedComponentsWithStats(candidate)
    # Stats already contain exact component areas. Background is always rejected,
    # so counting hundreds of thousands of background contacts is unnecessary.
    total=stats[:,cv2.CC_STAT_AREA]
    contact=np.bincount(labels[(candidate!=0)&(nearby_road!=0)],minlength=n)
    keep=(contact>=np.maximum(1,total*fraction))&(stats[:,cv2.CC_STAT_AREA]>=min_area)&(stats[:,cv2.CC_STAT_HEIGHT]>=8)
    keep[0]=False
    return np.uint8(keep[labels])*255


class GeometryPreprocessor:
    def __init__(self,base,geometry,marking):
        self.cfg={**deepcopy(base),**deepcopy(geometry['preprocessing'])}
        self.marking={**marking,**geometry['marking_overrides']}
        self.size=tuple(self.cfg['bev_size'])
        self.matrix=self.inverse=None;self.input_size=None

    def filter_candidate(self,candidate,bev,hsv,neutral):
        """Extension point for source-specific road context, before suppression."""
        return candidate

    def requires_initial_road_gate(self):
        """Generic preprocessing keeps the broad road-contact filter."""
        return True

    def process(self,frame):
        w=self.cfg['processing_width'];frame=cv2.resize(frame,(w,round(frame.shape[0]*w/frame.shape[1])))
        h,w=frame.shape[:2]
        anchors=np.float32(self.cfg['source_points'])*[w-1,h-1]
        polygon=np.float32(self.cfg['roi_polygon'])*[w-1,h-1]
        if self.input_size!=(w,h):
            dest=np.float32(self.cfg['destination_points'])*[self.size[0]-1,self.size[1]-1]
            self.matrix=cv2.getPerspectiveTransform(anchors.astype(np.float32),dest.astype(np.float32))
            self.inverse=np.linalg.inv(self.matrix);self.input_size=(w,h)
            self.roi_mask=np.zeros((h,w),np.uint8);cv2.fillPoly(self.roi_mask,[polygon.astype(np.int32)],255)
        roi=cv2.bitwise_and(frame,frame,mask=self.roi_mask)
        bev=cv2.warpPerspective(roi,self.matrix,self.size)
        hsv=cv2.cvtColor(bev,cv2.COLOR_BGR2HSV)
        blue,green,red=cv2.split(bev.astype(np.int16))
        field=(green-red>self.cfg.get('green_channel_excess',14))&(green-blue>5)
        neutral=np.uint8(~field)*255
        if self.cfg.get('mask_mode')=='white_thickness':
            white=cv2.inRange(hsv,(0,0,self.cfg['white_min_value']),(179,self.cfg['white_max_saturation'],255))
            white=cv2.bitwise_and(white,neutral)
            distance=cv2.distanceTransform(white,cv2.DIST_L2,3)
            seed=np.uint8(distance>=self.cfg['refined_min_half_thickness_px'])*255
            candidate=cv2.bitwise_and(white,cv2.dilate(seed,np.ones((7,7),np.uint8)))
        else:
            candidate=candidate_mask(bev,np.zeros(self.size[::-1],np.uint8),'refined',self.cfg)
        if self.requires_initial_road_gate():
            road=cv2.inRange(hsv,(0,0,20),(179,self.cfg['road_max_saturation'],self.cfg['road_max_value']))
            k=self.cfg['road_neighborhood'];road=cv2.dilate(road,np.ones((k,k),np.uint8))
            candidate=retain_road_adjacent_components(candidate,road,self.cfg['min_component_area'])
        candidate=cv2.bitwise_and(candidate,neutral)
        candidate=self.filter_candidate(candidate,bev,hsv,neutral)
        self.suppression=suppress_markings(bev,candidate,self.marking,hsv=hsv)
        self.paint=cv2.bitwise_and(self.suppression.raw_paint,neutral)
        self.candidate=candidate;self.anchor_points=anchors
        return Preprocessed(frame,roi,bev,self.suppression.binary,polygon)
