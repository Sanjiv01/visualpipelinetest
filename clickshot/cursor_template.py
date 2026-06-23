"""Learn this video's cursor appearance, then locate it every frame.

Motion tracking only sees the cursor while it MOVES, so "last position before the
change" lands mid-approach — the cursor has already stopped (and clicked) by then.
To pin the resting/click position we detect the cursor by appearance (template
matching), which finds it even when stationary.

Clean template extraction: when the cursor moves to a NEW position, the PREVIOUS
frame still shows the bare background there. Differencing the two frames at the new
position isolates the cursor exactly (foreground), and the new frame supplies its
colour. Averaging many such clean cut-outs (all centred on the cursor, so no
alignment needed) yields a sharp template + alpha mask.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import Config
from .models import VideoMeta


def _ignore_rects(cfg, w, h):
    return [(int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
            for (x0, y0, x1, y1) in cfg.ignore_regions]


def _in_rects(cx, cy, rects):
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for (x0, y0, x1, y1) in rects)


def learn_cursor_template(path: str, meta: VideoMeta, cfg: Config, target: int = 60):
    """Return (template_bgr, mask_uint8, (w, h), hotspot_xy) or None."""
    W, H = meta.width, meta.height
    diag = math.hypot(W, H)
    half = max(8, int(0.018 * diag))
    min_area = max(6, int(0.000015 * W * H))
    max_area = int(0.004 * W * H)
    jump = 0.20 * diag
    rects = _ignore_rects(cfg, W, H)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    step = max(1, round(meta.native_fps / max(1.0, meta.processed_fps)))

    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None

    crops, masks = [], []
    prev_f = prev_g = None
    last = vel = None
    idx = -1
    try:
        while len(crops) < target:
            ok, f = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if prev_g is None:
                prev_f, prev_g = f, g
                continue

            d = cv2.absdiff(prev_g, g)
            _, th = cv2.threshold(d, cfg.cursor_diff_thresh, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            nn, _, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
            cand = []
            for i in range(1, nn):
                a = stats[i, cv2.CC_STAT_AREA]
                bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                if a < min_area or a > max_area:
                    continue
                if max(bw, bh) / max(1, min(bw, bh)) > 3.5:
                    continue
                cx, cy = float(cents[i][0]), float(cents[i][1])
                if _in_rects(cx, cy, rects):
                    continue
                cand.append((cx, cy, int(bw), int(bh)))

            if cand:
                if last is not None and vel is not None:
                    pred = (last[0] + vel[0], last[1] + vel[1])
                    new = min(cand, key=lambda c: (c[0] - pred[0]) ** 2 + (c[1] - pred[1]) ** 2)
                else:
                    new = (min(cand, key=lambda c: c[2] * c[3]) if last is None
                           else min(cand, key=lambda c: (c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2))
                cx, cy, bw, bh = new
                if last is not None:
                    dist = math.hypot(cx - last[0], cy - last[1])
                    csize = max(bw, bh)
                    if dist < jump and csize * 1.4 < dist:   # clean separated move
                        crop = _clean_crop(f, prev_f, cx, cy, half, W, H)
                        if crop is not None:
                            crops.append(crop[0]); masks.append(crop[1])
                    vel = ((cx - last[0]) * 0.6 + (vel[0] if vel else 0) * 0.4,
                           (cy - last[1]) * 0.6 + (vel[1] if vel else 0) * 0.4)
                last = (cx, cy)
            prev_f, prev_g = f, g
    finally:
        cap.release()

    if len(crops) < 10:
        return None
    return _build(crops, masks)


def _clean_crop(f, pf, cx, cy, half, W, H):
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    if x0 < 0 or y0 < 0 or x0 + 2 * half >= W or y0 + 2 * half >= H:
        return None
    fg = f[y0:y0 + 2 * half, x0:x0 + 2 * half]
    bg = pf[y0:y0 + 2 * half, x0:x0 + 2 * half]
    diff = cv2.absdiff(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))
    mask = (diff > 22).astype(np.uint8) * 255
    return fg.copy(), mask


def _build(crops, masks):
    arr = np.stack(crops).astype(np.float32)
    msk = np.stack(masks).astype(np.float32) / 255.0
    coverage = msk.mean(axis=0)                       # how often each pixel is cursor
    mask = (coverage > 0.45).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    size = arr.shape[1]
    nn, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nn <= 1:
        return None
    c = size / 2
    cand = [(i, (cents[i][0] - c) ** 2 + (cents[i][1] - c) ** 2)
            for i in range(1, nn) if stats[i, cv2.CC_STAT_AREA] >= 8]
    if not cand:
        return None
    keep = min(cand, key=lambda t: t[1])[0]
    mask = np.where(lab == keep, 255, 0).astype(np.uint8)
    # colour each pixel from the crops where the cursor was actually present
    # (a plain median washes it out, since the cursor covers <50% of crops/pixel)
    count = msk.sum(axis=0)[..., None]
    template = ((arr * msk[..., None]).sum(axis=0) / np.maximum(count, 1)).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    tmpl = template[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    # hotspot = topmost-leftmost cursor pixel (the click tip for a standard pointer)
    hy = ys.min()
    hx = xs[ys == hy].min()
    hotspot = (int(hx - x0), int(hy - y0))
    return tmpl, m, (x1 - x0, y1 - y0), hotspot
