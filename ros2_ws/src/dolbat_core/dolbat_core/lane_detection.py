"""Histogram starts, sliding windows, robust quadratic fits and explicit validity."""
from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class LaneFit:
    coefficients: np.ndarray
    pixels: np.ndarray  # N x 2: image x, y
    residual_px: float
    coverage: float
    y_min: float
    y_max: float
    inferred: bool = False

    def x(self, y):
        return np.polyval(self.coefficients, y)


@dataclass
class LaneResult:
    left: LaneFit | None
    right: LaneFit | None
    status: str
    reason: str
    confidence: float
    debug: np.ndarray
    histogram: np.ndarray
    windows: list = field(default_factory=list)

    @property
    def usable(self):
        return self.status in ('DETECTED', 'INFERRED_LEFT', 'INFERRED_RIGHT')


def robust_fit(x, y, cfg, height):
    if len(x) < cfg['min_pixels_fit'] or np.ptp(y) < height * cfg['min_y_coverage']:
        return None
    keep = np.ones(len(x), dtype=bool)
    for _ in range(3):
        if np.count_nonzero(keep) < cfg['min_pixels_fit']:
            return None
        co = np.polyfit(y[keep] / (height - 1), x[keep], 2)
        residual = x - np.polyval(co, y / (height - 1))
        mad = np.median(np.abs(residual[keep] - np.median(residual[keep])))
        keep = np.abs(residual) < max(5.0, min(25.0, 3.0 * 1.4826 * mad))
    x, y = x[keep], y[keep]
    occupied = len(np.unique(np.minimum(y * cfg['windows'] // height, cfg['windows'] - 1)))
    if len(x) < cfg['min_pixels_fit'] or occupied < cfg['min_occupied_windows']:
        return None
    co = np.polyfit(y / (height - 1), x, 2) / np.array([(height - 1)**2, height - 1, 1])
    rms = float(np.sqrt(np.mean((x - np.polyval(co, y))**2)))
    coverage = float(np.ptp(y) / (height - 1))
    if rms > cfg['max_residual_px'] or coverage < cfg['min_y_coverage']:
        return None
    return LaneFit(co, np.column_stack((x, y)), rms, coverage, float(y.min()), float(y.max()))


class LaneDetector:
    def __init__(self, config):
        self.cfg = config
        self.previous = None
        self.missed = 0
        self.width = config['nominal_lane_width_px']

    def _slide(self, binary, start):
        h, w = binary.shape
        yy, xx = binary.nonzero()
        indices, windows = [], []
        current = float(start)
        last_center = None
        slope = 0.0
        for j in range(self.cfg['windows']):
            low, high = h - (j + 1) * h // self.cfg['windows'], h - j * h // self.cfg['windows']
            margin = self.cfg['margin_px']
            win = (int(current - margin), low, int(current + margin), high)
            windows.append(win)
            ids = np.flatnonzero((yy >= low) & (yy < high) & (np.abs(xx - current) <= margin))
            if len(ids) >= self.cfg['min_pixels_window']:
                # A median is less sensitive to side branches and small bright objects.
                center = float(np.median(xx[ids]))
                if last_center is not None:
                    slope = np.clip(center - last_center, -margin * .8, margin * .8)
                last_center = center
                current = center + slope
                indices.append(ids)
            else:
                current += slope
        if not indices:
            return None, windows
        ids = np.unique(np.concatenate(indices))
        return robust_fit(xx[ids], yy[ids], self.cfg, h), windows

    def detect(self, binary):
        h, w = binary.shape
        debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        histogram = np.count_nonzero(binary[int(h * .35):], axis=0).astype(float)
        smooth = np.convolve(histogram, np.ones(15) / 15, 'same')
        peaks = []
        remaining = smooth.copy()
        for _ in range(7):
            p = int(np.argmax(remaining))
            if remaining[p] < self.cfg['histogram_min_peak']:
                break
            peaks.append(p)
            sep = self.cfg['peak_separation_px']
            remaining[max(0, p-sep):p+sep+1] = 0
        if self.previous and self.missed <= self.cfg['max_missed_frames']:
            for lane in self.previous:
                peaks.insert(0, float(lane.x(h-1)))
        candidates = []
        for peak in peaks:
            fit, windows = self._slide(binary, peak)
            if fit is not None and not any(abs(fit.x(h*.75)-c[0].x(h*.75)) < 35 for c in candidates):
                candidates.append((fit, windows))
        best = None
        probes = np.linspace(h*.25, h-1, 15)
        for i, (a, aw) in enumerate(candidates):
            for b, bw in candidates[i+1:]:
                left, right, wins = (a, b, aw+bw) if a.x(h*.8) < b.x(h*.8) else (b, a, bw+aw)
                lo = max(left.y_min, right.y_min)
                hi = min(left.y_max, right.y_max)
                if hi - lo < h * .25:
                    continue
                widths = right.x(np.linspace(lo, hi, 15)) - left.x(np.linspace(lo, hi, 15))
                if min(widths) < self.cfg['width_range_px'][0] or max(widths) > self.cfg['width_range_px'][1]:
                    continue
                if np.ptp(widths) > self.cfg['max_width_variation_px']:
                    continue
                center_bottom = (left.x(h-1)+right.x(h-1))/2
                score = left.coverage + right.coverage - abs(np.median(widths)-self.width)/self.width
                score -= abs(center_bottom - w/2) / w
                if self.previous:
                    score -= np.mean([np.mean(abs(l.x(probes)-p.x(probes))) for l,p in zip((left,right),self.previous)])/80
                if best is None or score > best[0]:
                    best = (score, left, right, wins, float(np.median(widths)))
        status, reason = 'FAILED', 'no geometrically consistent lane pair'
        left = right = None
        windows = []
        if best:
            _, left, right, windows, width = best
            status, reason = 'DETECTED', 'two observed boundaries; geometry checks passed'
            self.width = .95 * self.width + .05 * width
        elif candidates:
            # Recover one observed boundary; synthetic partner is NEVER called DETECTED.
            ranked = []
            for fit, wins in candidates:
                for side in (0, 1):
                    reference = self.previous[side].x(probes) if self.previous else np.full_like(probes, w/2 + (side-.5)*self.width)
                    error = float(np.mean(abs(fit.x(probes)-reference)))
                    ranked.append((error - fit.coverage*15, error, side, fit, wins))
            _, error, side, fit, windows = min(ranked, key=lambda x:x[0])
            if error < self.cfg['max_temporal_jump_px'] * 1.5:
                co = fit.coefficients.copy()
                co[-1] += self.width if side == 0 else -self.width
                inferred = LaneFit(co, np.empty((0,2)), fit.residual_px, fit.coverage, fit.y_min, fit.y_max, True)
                left, right = (fit, inferred) if side == 0 else (inferred, fit)
                status = 'INFERRED_RIGHT' if side == 0 else 'INFERRED_LEFT'
                reason = 'one observed boundary; partner offset by assumed lane width'
        # Optional observation hook runs before temporal smoothing mutates coefficients.
        if hasattr(self, '_trace_selection'):
            self._trace_selection(left, right)
        if left is not None and self.previous:
            jump = max(float(np.mean(abs(l.x(probes)-p.x(probes)))) for l,p in zip((left,right),self.previous))
            if jump > self.cfg['max_temporal_jump_px']:
                left = right = None
                status, reason = 'REJECTED_JUMP', f'boundary displacement {jump:.1f}px exceeds limit'
            else:
                alpha = self.cfg['smoothing_alpha']
                for lane, old in zip((left,right),self.previous):
                    lane.coefficients = alpha*lane.coefficients + (1-alpha)*old.coefficients
        if left is None:
            self.missed += 1
            if self.missed > self.cfg['max_missed_frames']:
                self.previous = None
        else:
            self.previous = (left, right)
            self.missed = 0
        confidence = 0.0
        if left is not None:
            confidence = min(left.coverage, right.coverage) * np.exp(-max(left.residual_px,right.residual_px)/20)
            if status != 'DETECTED':
                confidence *= .5
            for lane, color in ((left,(255,160,0)),(right,(0,200,255))):
                if len(lane.pixels):
                    px = lane.pixels.astype(int)
                    debug[px[:,1], px[:,0]] = color
        for x1,y1,x2,y2 in windows:
            cv2.rectangle(debug,(x1,y1),(x2,y2),(50,120,50),1)
        return LaneResult(left,right,status,reason,float(confidence),debug,histogram,windows)
