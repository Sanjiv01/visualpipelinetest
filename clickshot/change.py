"""Meaningful screen-change detection.

Replaces the (provably inadequate) single global histogram with a recall-first
**max-fusion** of four spatial signals computed on a masked, downscaled, blurred
working frame:

1. pixel diff-ratio on luma AND Lab a/b chroma (catches color-only changes),
2. structural dissimilarity (1 - SSIM); the full SSIM map is reused for...
3. ...largest connected changed-component area, and
4. fraction of changed tiles (8x8 grid, Bhattacharyya histogram distance).

A persistent, *decaying* per-tile change-frequency EMA builds a dynamic-region
ignore mask so spinners / clocks / embedded video / blinking carets stop being
treated as change without permanently blinding those regions.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .config import Config
from .models import ChangeSignal, VideoMeta


def _blur_gray(bgr):
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 1)


class ChangeDetector:
    def __init__(self, cfg: Config, meta: VideoMeta):
        self.cfg = cfg
        self.h = meta.work_height
        self.w = meta.work_width
        self.area = float(self.h * self.w)
        self.g = cfg.tile_grid
        self.freq = np.zeros((self.g, self.g), np.float32)  # per-tile change EMA

    @property
    def dynamic_mask(self) -> np.ndarray:
        """uint8 mask, 0 where a tile is currently 'dynamic' (ignore), else 255."""
        m = np.full((self.h, self.w), 255, np.uint8)
        th, tw = self.h // self.g, self.w // self.g
        dyn = self.freq > self.cfg.dyn_freq_thresh
        for ty in range(self.g):
            for tx in range(self.g):
                if dyn[ty, tx]:
                    m[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw] = 0
        return m

    def score(self, prev_bgr, cur_bgr, valid_mask, *, update_freq=True) -> ChangeSignal:
        """Fused change score between two working frames under ``valid_mask``.

        ``valid_mask`` is uint8 (255 = compare). The detector's own dynamic-region
        mask is AND-ed in automatically.
        """
        cfg = self.cfg
        combined = cv2.bitwise_and(valid_mask, self.dynamic_mask)
        m = combined > 0
        valid = float(m.sum()) or 1.0

        # --- cheap prefilter over max BGR-channel diff (catches luma AND chroma).
        dmax = cv2.absdiff(prev_bgr, cur_bgr).max(axis=2)
        changed = (dmax > cfg.diff_pixel_thresh) & m
        diff_ratio = float(changed.sum()) / valid
        if diff_ratio < cfg.prefilter_ratio:
            return ChangeSignal(0, 0.0, 0.0, diff_ratio, 1.0, 0.0, 0.0, [])

        g0 = _blur_gray(prev_bgr)
        g1 = _blur_gray(cur_bgr)

        # --- (2) SSIM scalar + full map (reused for region analysis)
        s, full = ssim(g0, g1, full=True, data_range=255)
        change_map = ((1.0 - full) > cfg.ssim_map_thresh).astype(np.uint8)
        change_map[~m] = 0

        # --- (3) largest connected changed component
        n, _, stats, _ = cv2.connectedComponentsWithStats(change_map, connectivity=8)
        min_area = cfg.noise_area_frac * self.area
        regions, areas = [], []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a <= min_area:
                continue
            areas.append(a)
            regions.append((
                int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
            ))
        cc_frac = (max(areas) / self.area) if areas else 0.0

        # --- (4) tiled Bhattacharyya; update dynamic-region EMA
        tile_frac = self._tiles(g0, g1, update_freq)

        fused = min(1.0, max(
            diff_ratio / cfg.dr_ref,
            (1.0 - s) / cfg.ssim_ref,
            cc_frac / cfg.cc_ref,
            tile_frac,
        ))
        return ChangeSignal(0, 0.0, fused, diff_ratio, float(s), cc_frac, tile_frac, regions)

    def _tiles(self, g0, g1, update_freq) -> float:
        g = self.g
        th, tw = self.h // g, self.w // g
        changed_count = 0
        considered = 0
        dyn = self.freq > self.cfg.dyn_freq_thresh
        for ty in range(g):
            for tx in range(g):
                a = g0[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                b = g1[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                ha = cv2.normalize(cv2.calcHist([a], [0], None, [32], [0, 256]), None)
                hb = cv2.normalize(cv2.calcHist([b], [0], None, [32], [0, 256]), None)
                dist = cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)
                tch = dist > self.cfg.tile_bhat_thresh
                if update_freq:
                    self.freq[ty, tx] = (
                        (1 - self.cfg.dyn_ema_alpha) * self.freq[ty, tx]
                        + self.cfg.dyn_ema_alpha * float(tch)
                    )
                if dyn[ty, tx]:
                    continue  # dynamic tiles don't count toward a click consequence
                considered += 1
                changed_count += int(tch)
        return changed_count / considered if considered else 0.0


def calibrate_low_threshold(path: str, meta: VideoMeta, cfg: Config, samples: int = 60) -> float:
    """Per-video calm-threshold calibration from the LOWER tail of pair scores.

    The noise floor is the typical inter-frame delta during *static* stretches.
    We estimate it from a low percentile of evenly-spaced consecutive-pair scores
    (the static pairs), so real changes and unmasked cursor motion — which live in
    the upper tail — don't inflate it. The result is clamped to a sane band so it
    can never exceed the high threshold (which would disable detection).
    Deterministic (evenly spaced, not random) for reproducible runs.
    """
    import cv2 as _cv2

    det = ChangeDetector(cfg, meta)
    full_mask = np.full((meta.work_height, meta.work_width), 255, np.uint8)
    cap = _cv2.VideoCapture(path, _cv2.CAP_FFMPEG)
    scores = []
    try:
        n = meta.frame_count or 0
        if n < 4:
            return cfg.t_low
        points = np.linspace(2, n - 3, samples).astype(int)
        for p in points:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, int(p))
            ok0, f0 = cap.read()
            ok1, f1 = cap.read()
            if not (ok0 and ok1):
                continue
            w0 = _cv2.resize(f0, (meta.work_width, meta.work_height), interpolation=_cv2.INTER_AREA)
            w1 = _cv2.resize(f1, (meta.work_width, meta.work_height), interpolation=_cv2.INTER_AREA)
            scores.append(det.score(w0, w1, full_mask, update_freq=False).score)
    finally:
        cap.release()
    if not scores:
        return cfg.t_low
    noise = float(np.percentile(scores, 30))     # lower tail = the static pairs
    ceiling = 0.8 * cfg.t_high                    # never disable the state machine
    return float(min(ceiling, max(cfg.t_low, 1.5 * noise)))
