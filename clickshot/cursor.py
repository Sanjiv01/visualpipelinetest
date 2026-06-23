"""Cursor localization and the union ignore-mask.

Two complementary signals locate the cursor every frame at ORIGINAL resolution:

* the motion prior (the small, smoothly-moving diff blob) follows the cursor while
  it moves and seeds the search; and
* an appearance match against a template LEARNED from this video (cursor_template)
  pins the cursor even when STATIONARY — which is exactly when a click happens, so
  it captures the true resting/click position instead of a mid-approach point.

The appearance match multiplies a normalized EDGE response (flat regions have no
edges, so they can't false-match) with a masked COLOUR response (resolves the
y-ambiguity a lone edge leaves); a true cursor must score on both. Thresholds are
resolution-relative so this works across inputs, not just one video.

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


def _edges(gray) -> np.ndarray:
    """Sobel gradient magnitude (float32) — the matching feature for the cursor."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


class CursorTracker:
    """Stateful nearest-neighbour tracker over the motion prior.

    Keeps a last-known position so a *stationary* (dwelling) cursor — which emits
    no motion and so is invisible to the diff — is reported with speed ~0 rather
    than lost. That dwell signal is what click inference keys off.
    """

    def __init__(self, cfg: Config, meta: VideoMeta):
        self.cfg = cfg
        # track at ORIGINAL resolution — the cursor is too small once downscaled
        self.W, self.H = meta.width, meta.height
        self.diag = math.hypot(self.W, self.H)
        self.max_area = cfg.cursor_max_area_frac * self.W * self.H
        self.jump = cfg.cursor_jump_frac * self.diag
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.last_xy: tuple[float, float] | None = None
        self.last_wh: tuple[int, int] = (8, 8)
        self.prev_bbox: tuple[int, int, int, int] | None = None  # current frame's bbox
        self.detections = 0
        self.observations = 0
        self._miss = 0
        # optional learned cursor template (set via set_template) for appearance-based
        # detection that also finds the STATIONARY cursor at click time
        self.template = False
        self.match_thresh = 0.05   # combined edge*color score (product of two
                                   # normalized corrs, naturally small); higher = better
        self._prominence = 1.25    # best peak must beat runner-up by this factor
        self._vel = (0.0, 0.0)     # smoothed velocity (orig px/frame) for the velocity gate
        # Occupancy grid for dynamic-region exclusion. Cells are large enough that a
        # bounded oscillator (spinner/caret/video) stays inside one cell and trips the
        # dynamic flag, while a traversing cursor only visits each cell briefly.
        self._cell = max(40, int(self.diag / 8))
        self._cols = max(1, self.W // self._cell + 1)
        self._rows = max(1, self.H // self._cell + 1)
        self._motion_freq = np.zeros((self._rows, self._cols), np.float32)
        self._dynamic = np.zeros((self._rows, self._cols), bool)  # hysteretic exclusion
        self._miss_grace = max(3, round(0.6 * meta.processed_fps))
        # manual ignore rectangles in ORIGINAL pixels (e.g. a webcam overlay)
        self._ignore = [(x0 * self.W, y0 * self.H, x1 * self.W, y1 * self.H)
                        for (x0, y0, x1, y1) in cfg.ignore_regions]

    def set_template(self, tmpl, mask, size, hotspot):
        gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        m = mask > 0
        # EDGE-magnitude feature: flat/uniform regions have no edges and so can't
        # spuriously match (the failure mode of intensity correlation), while the
        # cursor's outline matches strongly. Edge alone gets x right but y ambiguous
        # (a lone arrow edge isn't unique), so it is GATED by a colour feature below.
        self._t_edge = _edges(gray)
        self._t_edge[~m] = 0
        # COLOUR feature: per-channel mean-subtracted template for TM_CCOEFF_NORMED,
        # which is discriminative (correlates shape of intensity variation, not raw
        # brightness) and resolves the y-ambiguity the edge map leaves. Masked pixels
        # are zeroed so background never contributes to the correlation.
        self._t_mask = m.astype(np.float32)
        self._mask_sum = float(self._t_mask.sum()) or 1.0
        col = tmpl.astype(np.float32)
        self._t_col = []
        for ch in range(3):
            c = col[:, :, ch] * self._t_mask
            mean = c.sum() / self._mask_sum
            c = (c - mean) * self._t_mask          # zero-mean over cursor pixels only
            self._t_col.append(c)
        # precompute template colour energy (for normalized cross-correlation)
        self._t_col_energy = math.sqrt(sum(float((c * c).sum()) for c in self._t_col)) or 1.0
        self._t_w, self._t_h = size
        self._hotspot = hotspot
        self.template = True

    def _color_response(self, roi_bgr):
        """Normalized colour cross-correlation map (peak ~1 at the cursor).

        Manual masked CCOEFF across the 3 BGR channels: for each candidate window we
        correlate the zero-mean template against the zero-mean (over cursor pixels)
        window, summed over channels, normalized by both energies. Masking means flat
        regions yield ~0 (no structured colour variation under the mask) instead of a
        spurious high score.
        """
        roi = np.ascontiguousarray(roi_bgr.astype(np.float32))
        num = None
        win_sq = None
        # accumulate numerator (cross-corr) and window energy over channels
        for ch in range(3):
            img = np.ascontiguousarray(roi[:, :, ch])
            # cross-correlation of (image) with (zero-mean masked template)
            cc = cv2.matchTemplate(img, self._t_col[ch], cv2.TM_CCORR)
            # local masked mean of the image: sum(image*mask)/mask_sum over window
            local_sum = cv2.matchTemplate(img, self._t_mask, cv2.TM_CCORR)
            local_mean = local_sum / self._mask_sum
            # because template is zero-mean over mask, sum(t)*local_mean == 0, so
            # cross-corr against the zero-mean window equals cc (the mean term cancels).
            num = cc if num is None else num + cc
            # window energy under mask: sum((img-localmean)^2 * mask)
            sum_i2 = cv2.matchTemplate(img * img, self._t_mask, cv2.TM_CCORR)
            energy = np.maximum(sum_i2 - self._mask_sum * (local_mean * local_mean), 0.0)
            win_sq = energy if win_sq is None else win_sq + energy
        denom = np.sqrt(win_sq) * self._t_col_energy
        resp = np.where(denom > 1e-6, num / np.maximum(denom, 1e-6), 0.0)
        resp[~np.isfinite(resp)] = 0.0
        return resp

    def _match(self, cur_gray, center, cur_bgr):
        """Locate the cursor by appearance near ``center``; return (cx, cy, score).

        edge-x-color: multiply the normalized EDGE response and the normalized COLOUR
        response. A true cursor must score on BOTH maps; flat regions (no edges) and
        lone-edge ambiguities (wrong colour structure) are both killed by the product.
        """
        R = int(max(self._t_w, self._t_h) * 1.5 + 0.08 * self.jump)
        cx, cy = center
        x0, y0 = max(0, int(cx - R)), max(0, int(cy - R))
        x1, y1 = min(self.W, int(cx + R)), min(self.H, int(cy + R))
        roi = cur_gray[y0:y1, x0:x1]
        if roi.shape[0] < self._t_h or roi.shape[1] < self._t_w:
            return None
        edge = cv2.matchTemplate(_edges(roi), self._t_edge, cv2.TM_CCORR_NORMED)
        edge[~np.isfinite(edge)] = 0.0
        color = self._color_response(cur_bgr[y0:y1, x0:x1])
        # both maps in [0,1]-ish; clip negatives so the product stays meaningful
        edge = np.clip(edge, 0.0, None)
        color = np.clip(color, 0.0, None)
        res = edge * color
        if res.size == 0:
            return None
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx < self.match_thresh:
            return None
        # peak-prominence: best must clearly beat the next, spatially-distinct peak,
        # otherwise the match is ambiguous (e.g. a repeated UI element) -> reject.
        if not self._prominent(res, loc):
            return None
        # The PRODUCT map gates out flat/ambiguous regions, but its peak is slightly
        # blurred by combining two features. The EDGE map alone localizes the cursor
        # outline more sharply, so refine the final coordinate to the edge peak within
        # a small neighbourhood of the gated product peak.
        loc = self._refine_to_edge(edge, loc)
        tlx, tly = x0 + loc[0], y0 + loc[1]      # template top-left in frame
        hx, hy = self._hotspot
        return (tlx + hx, tly + hy, mx)          # click point = top-left + hotspot

    def _refine_to_edge(self, edge, loc):
        """Snap to the strongest edge response within a few px of the product peak."""
        rad = 4
        x0 = max(0, loc[0] - rad); x1 = min(edge.shape[1], loc[0] + rad + 1)
        y0 = max(0, loc[1] - rad); y1 = min(edge.shape[0], loc[1] + rad + 1)
        sub = edge[y0:y1, x0:x1]
        if sub.size == 0:
            return loc
        dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        return (x0 + dx, y0 + dy)

    def _prominent(self, res, loc) -> bool:
        """True if the global peak beats the best peak outside its neighbourhood."""
        peak = res[loc[1], loc[0]]
        if peak <= 1e-6:
            return False
        masked = res.copy()
        rad = max(self._t_w, self._t_h)
        x0 = max(0, loc[0] - rad); x1 = min(res.shape[1], loc[0] + rad + 1)
        y0 = max(0, loc[1] - rad); y1 = min(res.shape[0], loc[1] + rad + 1)
        masked[y0:y1, x0:x1] = 0.0
        runner = float(masked.max())
        return runner <= 1e-6 or peak >= self._prominence * runner

    def _cell_of(self, cx, cy):
        return min(self._rows - 1, int(cy) // self._cell), min(self._cols - 1, int(cx) // self._cell)

    def _in_ignore(self, cx, cy):
        return any(x0 <= cx <= x1 and y0 <= cy <= y1 for (x0, y0, x1, y1) in self._ignore)

    def _invalidate_stale(self):
        """Drop a carried position that now sits in a dynamic/ignored region.

        Prevents an early webcam/animation latch from being carried forever once
        the real cursor stops being re-detected.
        """
        if self.last_xy is None:
            return
        r, c = self._cell_of(*self.last_xy)
        if self._dynamic[r, c] or self._in_ignore(*self.last_xy):
            self.last_xy = None

    def _detect_blobs(self, prev_gray, cur_gray):
        d = cv2.absdiff(prev_gray, cur_gray)
        _, th = cv2.threshold(d, self.cfg.cursor_diff_thresh, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, self._kernel)

        # Occupancy from ANY motion (vectorized over the cell grid) — large blobs
        # (a webcam face) mark their region dynamic even though they are too big to
        # be a cursor. This is what stops the cursor latching onto a webcam overlay.
        present = np.zeros((self._rows, self._cols), bool)
        ys, xs = np.nonzero(th)
        if xs.size:
            rr = np.minimum(ys // self._cell, self._rows - 1)
            cc = np.minimum(xs // self._cell, self._cols - 1)
            present[rr, cc] = True
        self._motion_freq = 0.92 * self._motion_freq + 0.08 * present
        # hysteresis: a cell becomes dynamic at >0.55, stays dynamic until <0.25
        self._dynamic = (self._motion_freq > 0.55) | (self._dynamic & (self._motion_freq > 0.25))

        n, _, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
        blobs = []
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
            if self._dynamic[r, c] or self._in_ignore(cx, cy):
                continue
            blobs.append((cx, cy, w, h, int(area), r, c))
        return blobs

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
        cur_gray = None
        if prev is not None:
            prev_gray = cv2.cvtColor(prev.orig, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(cur.orig, cv2.COLOR_BGR2GRAY)
            blobs = self._detect_blobs(prev_gray, cur_gray)
            self._invalidate_stale()   # drop a last position that now sits in a dynamic/ignored region
            cand = self._select(blobs)

        # appearance-based refinement: search near the motion hit (or the velocity-
        # predicted last position) so the cursor is located even when STATIONARY (i.e.
        # at the click moment). edge-x-color matching pins the resting cursor precisely.
        if self.template and cur_gray is not None:
            if cand is not None:
                center = (cand[0], cand[1])
            elif self.last_xy is not None:
                # predict with smoothed velocity so a fast-moving cursor's ROI keeps up
                center = (self.last_xy[0] + self._vel[0], self.last_xy[1] + self._vel[1])
            else:
                center = None
            m = None
            if center is not None and not self._in_ignore(*center):
                m = self._match(cur_gray, center, cur.orig)
            # full-frame re-acquire on loss: if no local match and we've been missing
            # for a while, search the whole frame around the motion candidate (or, if
            # none, skip — a full-frame scan with no prior is too ambiguous).
            if m is None and cand is not None and self.last_xy is not None:
                m = self._match(cur_gray, (cand[0], cand[1]), cur.orig)
            if m is not None:
                px, py, _ = m
                if self.last_xy is not None:
                    speed = math.hypot(px - self.last_xy[0], py - self.last_xy[1])
                    # velocity gate: a match that implies an implausible jump (and isn't
                    # backed by a near motion blob) is rejected as a false positive.
                    near_blob = cand is not None and math.hypot(
                        px - cand[0], py - cand[1]) <= max(self._t_w, self._t_h)
                    if speed > self.jump and not near_blob:
                        m = None
                if m is not None:
                    dx = px - self.last_xy[0] if self.last_xy is not None else 0.0
                    dy = py - self.last_xy[1] if self.last_xy is not None else 0.0
                    self._vel = (0.6 * dx + 0.4 * self._vel[0],
                                 0.6 * dy + 0.4 * self._vel[1])
                    speed = 0.0 if self.last_xy is None else math.hypot(dx, dy)
                    self.last_xy = (px, py)
                    self.last_wh = (self._t_w, self._t_h)
                    self._miss = 0
                    self.detections += 1
                    self.prev_bbox = _bbox(px, py, self._t_w, self._t_h)
                    return CursorObs(cur.index, cur.t, True, px, py, self._t_w, self._t_h, speed)

        if cand is not None:
            cx, cy, w, h = cand
            if self.last_xy is not None:
                dx, dy = cx - self.last_xy[0], cy - self.last_xy[1]
                speed = math.hypot(dx, dy)
                self._vel = (0.6 * dx + 0.4 * self._vel[0], 0.6 * dy + 0.4 * self._vel[1])
            else:
                speed = 0.0
            self.last_xy = (cx, cy)
            self.last_wh = (w, h)
            self._miss = 0
            self.detections += 1
            self.prev_bbox = _bbox(cx, cy, w, h)
            return CursorObs(cur.index, cur.t, True, cx, cy, w, h, speed)

        # No motion blob and no appearance match: cursor is dwelling or absent.
        # Decay the velocity prior so a stale ROI prediction does not run away.
        self._vel = (0.5 * self._vel[0], 0.5 * self._vel[1])
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
    """Bounding box of a cursor observation (original px), or None if unknown."""
    if o.x < 0:
        return None
    return _bbox(o.x, o.y, o.w, o.h)


def work_bbox_of(o: CursorObs, scale: float) -> tuple[int, int, int, int] | None:
    """Cursor bbox converted to working-frame coords (for the change mask)."""
    bb = bbox_of(o)
    if bb is None:
        return None
    x, y, w, h = bb
    return (int(x * scale), int(y * scale), max(1, int(w * scale)), max(1, int(h * scale)))


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
