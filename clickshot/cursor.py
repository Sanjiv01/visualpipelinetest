"""Cursor localization (motion-prior) and the union ignore-mask.

On compressed, low-resolution screen video the cursor is tiny (~11-20 px at
640x360), so template matching alone is unreliable. The *primary* localizer is
therefore the motion prior: the small, smoothly-moving connected component in the
frame-to-frame diff. (A pywin32 template *shape classifier* can be layered on top
later; it is not required for masking.)

The single most important correctness detail in the whole system lives here:
``union_mask`` masks the dilated UNION of the cursor footprint in frame n AND
frame n+1. Masking only the current position leaves a phantom "change" where the
cursor *used to be* — a moving cursor reveals the pixels it previously occluded.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import Config
from .models import CursorObs, FrameRecord, VideoMeta


class CursorTracker:
    """Stateful nearest-neighbour tracker over the motion prior.

    Keeps a last-known position so a *stationary* (dwelling) cursor — which emits
    no motion and so is invisible to the diff — is reported with speed ~0 rather
    than lost. That dwell signal is what click inference keys off.
    """

    def __init__(self, cfg: Config, meta: VideoMeta):
        self.cfg = cfg
        self.diag = math.hypot(meta.work_width, meta.work_height)
        self.max_area = cfg.cursor_max_area_frac * meta.work_width * meta.work_height
        self.jump = cfg.cursor_jump_frac * self.diag
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.last_xy: tuple[float, float] | None = None
        self.last_wh: tuple[int, int] = (8, 8)
        self.prev_bbox: tuple[int, int, int, int] | None = None  # current frame's bbox
        self.detections = 0
        self.observations = 0
        self._miss = 0
        # Occupancy grid for dynamic-region exclusion. Cells are large enough that a
        # bounded oscillator (spinner/caret/video) stays inside one cell and trips the
        # dynamic flag, while a traversing cursor only visits each cell briefly.
        self._cell = max(40, int(self.diag / 8))
        self._cols = max(1, meta.work_width // self._cell + 1)
        self._rows = max(1, meta.work_height // self._cell + 1)
        self._motion_freq = np.zeros((self._rows, self._cols), np.float32)
        self._miss_grace = max(3, round(0.6 * meta.processed_fps))

    def _cell_of(self, cx, cy):
        return min(self._rows - 1, int(cy) // self._cell), min(self._cols - 1, int(cx) // self._cell)

    def _detect_blobs(self, prev_gray, cur_gray):
        d = cv2.absdiff(prev_gray, cur_gray)
        _, th = cv2.threshold(d, self.cfg.cursor_diff_thresh, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, self._kernel)
        n, _, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)

        blobs = []
        present = np.zeros((self._rows, self._cols), bool)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < self.cfg.cursor_min_area or area > self.max_area:
                continue
            if max(w, h) / max(1, min(w, h)) > self.cfg.cursor_max_aspect:
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            r, c = self._cell_of(cx, cy)
            present[r, c] = True
            blobs.append((cx, cy, w, h, int(area), r, c))

        self._motion_freq *= (1 - 0.15)
        self._motion_freq[present] += 0.15
        dynamic = self._motion_freq > 0.5
        return [b for b in blobs if not dynamic[b[5], b[6]]]

    def _select(self, cands):
        """Return (cx, cy, w, h) or None (carry/dwell)."""
        if self.last_xy is not None and cands:
            in_jump = [(math.hypot(b[0] - self.last_xy[0], b[1] - self.last_xy[1]), b)
                       for b in cands]
            in_jump = [t for t in in_jump if t[0] <= self.jump]
            if in_jump:
                b = min(in_jump, key=lambda t: t[0] + 0.5 * t[1][4])[1]
                return (b[0], b[1], b[2], b[3])
        # No near candidate. During a short gap keep dwelling on the last position
        # (a stationary cursor emits no motion) rather than jumping to a far blob.
        if self.last_xy is not None and self._miss < self._miss_grace:
            return None
        if cands:  # re-acquire: the smallest fresh, non-dynamic blob is the cursor
            b = min(cands, key=lambda b: b[4])
            return (b[0], b[1], b[2], b[3])
        return None

    def update(self, prev: FrameRecord | None, cur: FrameRecord) -> CursorObs:
        self.observations += 1
        cand = None
        if prev is not None:
            prev_gray = cv2.cvtColor(prev.work, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(cur.work, cv2.COLOR_BGR2GRAY)
            cand = self._select(self._detect_blobs(prev_gray, cur_gray))

        if cand is not None:
            cx, cy, w, h = cand
            speed = 0.0 if self.last_xy is None else math.hypot(
                cx - self.last_xy[0], cy - self.last_xy[1]
            )
            self.last_xy = (cx, cy)
            self.last_wh = (w, h)
            self._miss = 0
            self.detections += 1
            self.prev_bbox = _bbox(cx, cy, w, h)
            return CursorObs(cur.index, cur.t, True, cx, cy, w, h, speed)

        # No motion blob: cursor is either dwelling (stationary) or absent.
        self._miss += 1
        if self.last_xy is not None:
            cx, cy = self.last_xy
            w, h = self.last_wh
            self.prev_bbox = _bbox(cx, cy, w, h)
            return CursorObs(cur.index, cur.t, False, cx, cy, w, h, 0.0)
        self.prev_bbox = None
        return CursorObs(cur.index, cur.t, False, -1.0, -1.0, 0, 0, 0.0)

    @property
    def detection_rate(self) -> float:
        return self.detections / self.observations if self.observations else 0.0


def _bbox(cx: float, cy: float, w: int, h: int) -> tuple[int, int, int, int]:
    return (int(round(cx - w / 2)), int(round(cy - h / 2)), max(1, w), max(1, h))


def bbox_of(o: CursorObs) -> tuple[int, int, int, int] | None:
    """Bounding box of a cursor observation, or None if position is unknown."""
    if o.x < 0:
        return None
    return _bbox(o.x, o.y, o.w, o.h)


def union_mask(shape, bbox_a, bbox_b, cfg: Config) -> np.ndarray:
    """Return uint8 mask: 255 = VALID (compare), 0 = IGNORE (cursor footprint).

    Masks the dilated union of both bounding boxes. ``shape`` is (H, W[, C]).
    """
    h, w = shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    m = cfg.mask_margin_px
    for bb in (bbox_a, bbox_b):
        if bb is None:
            continue
        x, y, bw, bh = bb
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(w, x + bw + m), min(h, y + bh + m)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    return mask
