"""Reject changes that are not click consequences.

Two filters run over each candidate event:

* **revert / transient** — if the stabilized frame is ~identical to the
  pre-transition reference, the screen returned to where it started (hover
  highlight, tooltip, transient popover): not a consequence.
* **scroll** — coherent, large-area, vertical translation (dense optical flow
  cross-checked by phase correlation). The dominant false positive in screen
  recordings. A scroll is only rejected when there is no strong click signal,
  so a click that *triggers* an animated slide is not thrown away.
"""

from __future__ import annotations

import cv2
import numpy as np

from .change import ChangeDetector
from .config import Config
from .models import Event, VideoMeta


def scroll_signature(prev_gray, next_gray, cfg: Config):
    """Per-pair scroll metrics, or None if there isn't coherent vertical translation."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    moving = mag > 1.0
    if moving.mean() < cfg.scroll_flow_area:
        return None
    mdy = float(np.median(fy[moving]))
    mdx = float(np.median(fx[moving]))
    if abs(mdy) < cfg.scroll_min_shift or abs(mdy) <= 2 * abs(mdx):
        return None
    aligned = (np.sign(fy[moving]) == np.sign(mdy)) & \
              (np.abs(fy[moving] - mdy) < 0.4 * abs(mdy) + 1e-3)
    coherence = float(aligned.mean())
    if coherence < cfg.scroll_coherence:
        return None
    return {"dy": mdy, "area": float(moving.mean()), "coherence": coherence}


def _gray(work):
    return cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)


def _scroll_verdict(span, cfg: Config):
    """Scroll if a majority of adjacent span pairs show coherent vertical translation."""
    pairs = max(1, len(span) - 1)
    hits = [scroll_signature(_gray(span[i][1]), _gray(span[i + 1][1]), cfg)
            for i in range(len(span) - 1)]
    hits = [h for h in hits if h]
    if len(hits) / pairs < cfg.scroll_pair_frac:
        return None
    mdy = np.median([h["dy"] for h in hits])
    area = np.median([h["area"] for h in hits])
    return {"why": f"coherent vertical translation dy~{mdy:.1f}px over "
                   f"{area:.0%} of frame in {len(hits)}/{pairs} steps"}


def apply(events: list[Event], cfg: Config, meta: VideoMeta) -> list[Event]:
    net_det = ChangeDetector(cfg, meta)  # fresh (no dynamic mask) for clean net change
    full = np.full((meta.work_height, meta.work_width), 255, np.uint8)

    for ev in events:
        tr = ev.transition
        if tr.ref_work is None or tr.consequence_work is None:
            continue

        net = net_det.score(tr.ref_work, tr.consequence_work, full, update_freq=False)
        if net.score < cfg.net_change_min:
            ev.accepted = False
            ev.rejected.append({"type": "revert", "t": tr.start_t,
                                "why": "consequence ~= pre-click state (hover/transient)"})
            continue

        scroll = _scroll_verdict(tr.span_works, cfg)
        if scroll:
            strong_click = ev.click is not None and ev.confidence >= 0.55
            if not strong_click:
                ev.accepted = False
                ev.rejected.append({"type": "scroll", "t": tr.start_t, "why": scroll["why"]})
                continue
            ev.reasons.append("scroll-like motion present but a click was detected")

    return events
