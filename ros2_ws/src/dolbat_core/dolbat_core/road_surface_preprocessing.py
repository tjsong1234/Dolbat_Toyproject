"""Keep paint next to the connected road surface instead of background shadows."""
import cv2
import numpy as np
from .geometry_preprocessing import GeometryPreprocessor,retain_road_adjacent_components


class RoadSurfacePreprocessor(GeometryPreprocessor):
    def requires_initial_road_gate(self):
        # In white_thickness mode candidate is already a subset of neutral white.
        # The later neutral bitwise-and cannot split its components. Connected
        # contact road is a subset of this initial broad road if its saturation
        # bound is no larger. Equal-area component gates with the same threshold
        # are then dominated by the later, stricter gate, so the first CC pass is
        # redundant. Preserve generic/legacy behavior whenever this proof fails.
        return not (self.cfg.get('mask_mode')=='white_thickness'
                    and self.cfg.get('surface_max_saturation',70)<=self.cfg['road_max_saturation'])

    def filter_candidate(self,candidate,bev,hsv,neutral):
        road=cv2.inRange(hsv,(0,0,20),(179,self.cfg.get('surface_max_saturation',70),self.cfg['road_max_value'])) & neutral
        white=cv2.inRange(hsv,(0,0,self.cfg['white_min_value']),(179,self.cfg['white_max_saturation'],255)) & neutral
        # Paint may divide a road at a stop line; admit actual paint for connectivity.
        surface=road | white
        n,labels,stats,_=cv2.connectedComponentsWithStats(surface)
        scores=np.bincount(labels[int(bev.shape[0]*.55):].ravel(),minlength=n).astype(float)
        scores[0]=0
        self.road_surface=np.zeros_like(road)
        if scores.max()>self.cfg.get('surface_min_seed_area',400):
            chosen=int(np.argmax(scores));self.road_surface=np.uint8(labels==chosen)*255
            contact_road=road & self.road_surface
            k=self.cfg['road_neighborhood']
            nearby=cv2.dilate(contact_road,np.ones((k,k),np.uint8))
            candidate=retain_road_adjacent_components(candidate,nearby,self.cfg['min_component_area'])
        else:
            # Do not silently claim a disconnected background component is road.
            candidate=np.zeros_like(candidate)
        return candidate
