"""Perceptual-hash dedup of near-identical consequence frames.

Dependency-free DCT pHash (cv2.dct is in base OpenCV). Two events are merged only
when they are close in TIME and in pHash — never across distant timestamps — so
genuinely distinct screens that happen to look alike are not collapsed. The
sharper frame survives.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .models import Event


def phash(work) -> int:
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8].flatten()
    med = float(np.median(low[1:]))  # exclude DC term
    bits = low > med
    val = 0
    for b in bits[1:]:               # drop DC bit
        val = (val << 1) | int(bool(b))
    return val


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def run(events: list[Event], cfg: Config) -> list[Event]:
    for ev in events:
        if ev.transition.consequence_work is not None:
            ev.phash = phash(ev.transition.consequence_work)

    kept: list[Event] = []
    for ev in events:
        if not ev.accepted:
            continue
        dup_of = None
        for prev in reversed(kept):
            if ev.transition.consequence_t - prev.transition.consequence_t > cfg.dedup_min_gap_s:
                break
            if hamming(ev.phash, prev.phash) <= cfg.dedup_hamming:
                dup_of = prev
                break
        if dup_of is None:
            kept.append(ev)
            continue
        # keep the sharper frame
        if ev.transition.sharpness > dup_of.transition.sharpness:
            dup_of.accepted = False
            dup_of.dup_of = ev.id
            kept.remove(dup_of)
            kept.append(ev)
        else:
            ev.accepted = False
            ev.dup_of = dup_of.id
    return events
