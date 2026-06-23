"""Per-stage unit tests."""

from __future__ import annotations

import cv2
import numpy as np

from clickshot import cursor, dedup
from clickshot.change import ChangeDetector
from clickshot.config import Config
from clickshot.models import VideoMeta


def _meta(w, h):
    return VideoMeta("x", w, h, 30.0, 100, 12.0, w, h, 1.0)


def _full(w, h):
    return np.full((h, w), 255, np.uint8)


def test_change_identical_is_zero():
    det = ChangeDetector(Config(), _meta(200, 120))
    a = np.full((120, 200, 3), 50, np.uint8)
    sig = det.score(a, a.copy(), _full(200, 120), update_freq=False)
    assert sig.score < 0.05


def test_change_panel_swap_is_high():
    det = ChangeDetector(Config(), _meta(200, 120))
    a = np.full((120, 200, 3), 50, np.uint8)
    b = a.copy()
    cv2.rectangle(b, (20, 20), (180, 100), (200, 180, 120), -1)
    sig = det.score(a, b, _full(200, 120), update_freq=False)
    assert sig.score > 0.5


def test_change_color_only_swap_detected():
    """A same-luminance color change must still register (Lab chroma signal)."""
    det = ChangeDetector(Config(), _meta(200, 120))
    a = np.full((120, 200, 3), (120, 120, 120), np.uint8)
    b = np.full((120, 200, 3), (120, 60, 200), np.uint8)  # different chroma
    sig = det.score(a, b, _full(200, 120), update_freq=False)
    assert sig.score > 0.3


def test_dedup_phash_identical():
    a = (np.arange(120 * 200 * 3) % 255).astype(np.uint8).reshape(120, 200, 3)
    assert dedup.hamming(dedup.phash(a), dedup.phash(a.copy())) == 0


def test_union_mask_covers_both_positions():
    cfg = Config()
    m = cursor.union_mask((100, 100, 3), (10, 10, 8, 8), (60, 60, 8, 8), cfg)
    assert m[14, 14] == 0       # inside first cursor footprint -> ignored
    assert m[64, 64] == 0       # inside second cursor footprint -> ignored
    assert m[40, 40] == 255     # between them -> valid


def test_bbox_of_unknown_is_none():
    from clickshot.models import CursorObs
    assert cursor.bbox_of(CursorObs(0, 0.0, False, -1, -1, 0, 0, 0.0)) is None


def test_scroll_signature_detects_vertical_shift():
    from clickshot import filters
    rng = np.random.default_rng(1)
    base = cv2.resize(rng.integers(0, 255, (60, 80, 3), np.uint8), (400, 300),
                      interpolation=cv2.INTER_LINEAR)
    g0 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(np.roll(base, 7, axis=0), cv2.COLOR_BGR2GRAY)
    sig = filters.scroll_signature(g0, g1, Config())
    assert sig is not None and abs(sig["dy"]) > 2  # coherent vertical translation found
