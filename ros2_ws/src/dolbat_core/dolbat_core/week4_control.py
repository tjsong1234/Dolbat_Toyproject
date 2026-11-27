"""Offline Pure Pursuit on a forward-ordered path in rear-axle coordinates.

X points forward and Y points left. Distances are metres; positive steering
means a left turn. The target is an actual polyline/circle intersection, not
the path point whose X happens to equal the lookahead radius. No extrapolated
target, steering clamp, or temporal filter is applied here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_m: float = 1.7
    wheelbase_m: float = 0.55

    def __post_init__(self):
        for name in ('lookahead_m', 'wheelbase_m'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class PurePursuitResult:
    valid: bool
    status: str
    target_m: tuple[float, float] | None = None
    curvature_per_m: float | None = None
    steering_rad: float | None = None
    steering_deg: float | None = None
    heading_error_rad: float | None = None
    segment_index: int | None = None
    segment_fraction: float | None = None
    circle_residual_m: float | None = None
    candidate_count: int = 0
    actual_lookahead_m: float | None = None


def bev_cm_to_rear_m(points_cm, forward_offset_m: float, lateral_offset_m: float = 0.0) -> np.ndarray:
    """Translate the BEV reference into the rear-axle frame, without mutation.

    ``forward_offset_m`` is the rear-axle-frame forward coordinate of the BEV
    reference origin. It must be calibrated for the particular homography;
    it is not automatically the distance to the original camera image bottom.
    ``lateral_offset_m`` translates the BEV origin onto the assumed vehicle
    centreline. There is no rotation; the rear-frame heading must be aligned
    with BEV forward X by the caller's geometric calibration.
    """
    points = np.asarray(points_cm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('points_cm must be a finite Nx2 array')
    if not math.isfinite(forward_offset_m) or not math.isfinite(lateral_offset_m):
        raise ValueError('frame offsets must be finite')
    result = points / 100.0
    result[:, 0] += forward_offset_m
    result[:, 1] += lateral_offset_m
    return result


def _invalid(status: str) -> PurePursuitResult:
    return PurePursuitResult(False, status)


def calculate_pure_pursuit(path_rear_m, config: PurePursuitConfig | None = None) -> PurePursuitResult:
    """Find the first forward circle intersection in path traversal order.

    The input is a finite Nx2 polyline ordered by nondecreasing forward X.
    Shared-vertex intersections are counted once. For a path that enters and
    leaves the circle several times, the first positive-X crossing is selected;
    ``candidate_count`` records all distinct forward intersections. Invalid or
    insufficient paths and missing intersections return an explicit failed
    result with no angle rather than replaying the previous frame's command.
    """
    config = config or PurePursuitConfig()
    if path_rear_m is None:
        return _invalid('NO_PATH')
    try:
        points = np.asarray(path_rear_m, dtype=float)
    except (TypeError, ValueError):
        return _invalid('INVALID_PATH')
    if points.ndim != 2 or points.shape[1] != 2:
        return _invalid('INVALID_PATH')
    if len(points) < 2:
        return _invalid('NO_PATH')
    if not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) < -1e-10):
        return _invalid('INVALID_PATH')

    radius = config.lookahead_m
    candidates: list[tuple[int, float, np.ndarray]] = []
    nonzero_segment = False
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        delta = end - start
        aa = float(delta @ delta)
        if aa <= 1e-24:
            continue
        nonzero_segment = True
        bb = 2.0 * float(start @ delta)
        cc = float(start @ start) - radius * radius
        disc = bb * bb - 4.0 * aa * cc
        tolerance = 64 * np.finfo(float).eps * (bb * bb + abs(4.0 * aa * cc) + 1.0)
        if disc < -tolerance:
            continue
        root = math.sqrt(max(0.0, disc))
        # The q formulation avoids cancellation when one quadratic root is tiny.
        q = -0.5 * (bb + math.copysign(root, bb))
        roots = [-bb / (2.0 * aa)] if abs(q) < 1e-30 else [q / aa, cc / q]
        for fraction in sorted(roots):
            if fraction < -1e-10 or fraction > 1.0 + 1e-10:
                continue
            fraction = float(np.clip(fraction, 0.0, 1.0))
            target = start + fraction * delta
            if target[0] <= 1e-10:
                continue
            if any(np.linalg.norm(target - previous[2]) <= 1e-9 for previous in candidates):
                continue
            candidates.append((index, fraction, target))
    if not nonzero_segment:
        return _invalid('DEGENERATE_PATH')
    if not candidates:
        return _invalid('NO_FORWARD_INTERSECTION')

    index, fraction, target = candidates[0]
    x, y = map(float, target)
    distance_squared = x * x + y * y
    distance = math.sqrt(distance_squared)
    curvature = 2.0 * y / distance_squared
    steering = math.atan(config.wheelbase_m * curvature)
    return PurePursuitResult(
        valid=True, status='OK', target_m=(x, y), curvature_per_m=curvature,
        steering_rad=steering, steering_deg=math.degrees(steering),
        heading_error_rad=math.atan2(y, x), segment_index=index,
        segment_fraction=fraction, circle_residual_m=abs(distance - radius),
        candidate_count=len(candidates), actual_lookahead_m=distance,
    )
