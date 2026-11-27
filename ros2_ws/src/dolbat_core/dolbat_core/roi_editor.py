"""Small, testable geometry and persistence helpers for the OpenCV ROI player."""
from datetime import datetime
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml


def order_quad(points):
    """Assign four upright road-ROI corners to TL, TR, BR, BL, regardless of click order."""
    p = np.asarray(points,dtype=float)
    if p.shape != (4,2) or not np.isfinite(p).all():
        raise ValueError('Four finite corner coordinates are required')
    by_height = p[np.argsort(p[:,1],kind='stable')]
    top = by_height[:2][np.argsort(by_height[:2,0])]
    bottom = by_height[2:][np.argsort(by_height[2:,0])]
    return np.array([top[0],top[1],bottom[1],bottom[0]])


def validate_quad(points, width, height):
    """Ordered TL, TR, BR, BL, convex clockwise in image coordinates."""
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (4, 2):
        return False, 'Select all 4 ROI corners before saving'
    if not np.isfinite(p).all():
        return False, 'Coordinates must be finite'
    if np.any(p[:, 0] < 0) or np.any(p[:, 0] > width-1) or np.any(p[:, 1] < 0) or np.any(p[:, 1] > height-1):
        return False, 'All corners must be inside the original image'
    edges = np.roll(p, -1, axis=0) - p
    crosses = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
    if np.any(crosses <= 0) or cv2.contourArea(p) < width*height*.005:
        return False, 'ROI must be a convex quadrilateral: TL > TR > BR > BL'
    if (p[0, 1]+p[1, 1]) >= (p[2, 1]+p[3, 1]) or p[0,0] >= p[1,0] or p[3,0] >= p[2,0]:
        return False, 'Use corner order TL > TR > BR > BL'
    return True, 'ROI valid - drag corners while playing or paused'


def make_bev(frame, points, size):
    h, w = frame.shape[:2]
    valid, message = validate_quad(points, w, h)
    if not valid:
        return None, message
    bw, bh = size
    dest = np.float32([[0,0],[bw-1,0],[bw-1,bh-1],[0,bh-1]])
    matrix = cv2.getPerspectiveTransform(np.float32(points), dest)
    return cv2.warpPerspective(frame, matrix, (bw,bh)), message


def save_roi_config(config_path, points, source_size, video_path, frame_index):
    """Save normalized points; preserve all other settings and back up the old file."""
    config_path = Path(config_path)
    w, h = source_size
    valid, message = validate_quad(points,w,h)
    if not valid:
        raise ValueError(message)
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    normalized = np.asarray(points,dtype=float)/[w-1,h-1]
    cfg['preprocessing']['source_points'] = normalized.tolist()
    # Full BEV rectangle is the editor's live destination, so saved and previewed transforms agree.
    cfg['preprocessing']['destination_points'] = [[0,0],[1,0],[1,1],[0,1]]
    cfg['roi_selection'] = {
        'video': Path(video_path).name,
        'frame_index_zero_based': int(frame_index),
        'source_size': [int(w),int(h)],
        'corner_order': 'TL, TR, BR, BL',
        'selected_at': datetime.now().astimezone().isoformat(),
        'metric_calibration': 'ROI selection does not calibrate physical distances',
    }
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = config_path.with_name(f'{config_path.stem}.before_roi_{stamp}{config_path.suffix}')
    shutil.copy2(config_path,backup)
    temp = config_path.with_suffix(config_path.suffix+'.tmp')
    temp.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    temp.replace(config_path)
    return backup
