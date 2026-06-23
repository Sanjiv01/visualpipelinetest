"""Plain data containers passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VideoMeta:
    path: str
    width: int           # native width
    height: int          # native height
    native_fps: float
    frame_count: int
    processed_fps: float  # effective fps after subsampling
    work_width: int       # working (downscaled) width
    work_height: int      # working (downscaled) height
    scale: float          # work / native


@dataclass
class FrameRecord:
    """One sampled frame. ``work`` is the downscaled BGR image used for all CV."""

    index: int            # native frame index (for re-reading the original on output)
    t: float              # seconds from start
    work: np.ndarray      # downscaled BGR frame


@dataclass
class CursorObs:
    index: int
    t: float
    found: bool           # was the cursor blob detected this frame
    x: float              # working-frame coords (last known if not found)
    y: float
    w: int
    h: int
    speed: float          # working px moved since previous observation


@dataclass
class ChangeSignal:
    index: int
    t: float
    score: float          # fused, recall-first change score in [0, 1]
    diff_ratio: float
    ssim: float
    cc_frac: float        # largest changed component, fraction of frame area
    tile_frac: float      # fraction of tiles changed
    regions: list = field(default_factory=list)  # (x, y, w, h) changed CCs, working coords


@dataclass
class Transition:
    start_t: float
    end_t: float
    start_index: int
    end_index: int
    peak_score: float
    min_ssim: float
    timeout: bool
    changed_regions: list
    # consequence selection (filled by the state machine):
    consequence_index: int
    consequence_t: float
    sharpness: float
    # retained imagery (working frames) for downstream filters / dedup / output:
    ref_work: np.ndarray | None = None          # last stable frame BEFORE the transition
    ref_index: int = -1
    ref_t: float = 0.0
    consequence_work: np.ndarray | None = None  # the chosen stable frame (working res)
    span_works: list = field(default_factory=list)  # (t, work) sampled across the span


@dataclass
class ClickEvent:
    t: float
    x: float              # working-frame coords
    y: float
    button: str
    source: str           # 'inferred' (video-only) — kept for parity with a future precise mode


@dataclass
class Event:
    id: int
    transition: Transition
    click: ClickEvent | None
    confidence: float
    reasons: list
    rejected: list        # [{type, t, why}] candidates filtered out near this transition
    phash: int
    dup_of: int | None = None
    accepted: bool = True
