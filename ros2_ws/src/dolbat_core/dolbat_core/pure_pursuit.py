"""Offline bicycle-model steering, positive left, negative right."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ControlResult:
    status: str
    target: np.ndarray | None
    alpha_rad: float | None
    actual_lookahead_m: float | None
    curvature_per_m: float | None
    raw_steering_deg: float | None
    steering_deg: float | None


def world_to_vehicle(points, position=(0.,0.), heading_rad=0.):
    points = np.asarray(points, dtype=float) - np.asarray(position)
    c,s = math.cos(heading_rad),math.sin(heading_rad)
    return points @ np.array([[c,-s],[s,c]])


def select_target(path, lookahead):
    """First forward intersection of path segments with the lookahead circle.

    No extrapolation: insufficient visible path returns None rather than a fake target.
    """
    for a,b in zip(path[:-1],path[1:]):
        d = b-a
        aa = float(d@d)
        if aa < 1e-12:
            continue
        bb,cc = 2*float(a@d),float(a@a)-lookahead**2
        disc = bb*bb-4*aa*cc
        if disc < 0:
            continue
        for t in sorted(((-bb-math.sqrt(disc))/(2*aa),(-bb+math.sqrt(disc))/(2*aa))):
            target = a+t*d
            if -1e-9 <= t <= 1+1e-9 and target[0] > 0:
                return target
    return None


class PurePursuit:
    def __init__(self, config):
        self.cfg = config

    def compute(self, path, position=(0.,0.), heading_rad=0.):
        if path is None:
            return ControlResult('NO_PATH',None,None,None,None,None,None)
        local = world_to_vehicle(path,position,heading_rad)
        target = select_target(local,self.cfg['lookahead_m'])
        if target is None:
            return ControlResult('NO_LOOKAHEAD_INTERSECTION',None,None,None,None,None,None)
        ld = float(np.linalg.norm(target))
        alpha = math.atan2(target[1],target[0])
        kappa = 2*math.sin(alpha)/ld
        raw = math.degrees(math.atan(self.cfg['wheelbase_m']*kappa))
        clipped = float(np.clip(raw,-self.cfg['max_steering_deg'],self.cfg['max_steering_deg']))
        return ControlResult('SATURATED' if abs(raw-clipped)>1e-6 else 'OK',target,alpha,ld,kappa,raw,clipped)
