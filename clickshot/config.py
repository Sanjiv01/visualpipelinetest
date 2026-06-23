"""Single source of truth for every tunable threshold.

A frozen ``Config`` is threaded through every pipeline stage and serialized into
``events.json`` so any run is reproducible. Defaults are resolution-independent
(fractions / seconds, never raw pixel counts at a fixed resolution) so one config
works on the 640x360 sample and on a 4K capture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    # --- frame i/o -------------------------------------------------------
    sample_fps: float = 24.0          # process this many frames/sec (recall-first; raise to taste)
    work_long_side: int = 480         # downscale long side for all CV work

    # --- cursor detection / masking -------------------------------------
    cursor_diff_thresh: int = 22      # abs-diff threshold to find the moving blob
    cursor_min_area: int = 3          # min blob area (working px) to be a cursor
    cursor_max_area_frac: float = 0.02  # max blob area as fraction of working frame
    cursor_max_aspect: float = 3.5    # reject very elongated blobs (scroll strips)
    cursor_jump_frac: float = 0.25    # max plausible per-frame cursor move (frac of diag)
    mask_margin_px: int = 10          # safety margin added around the cursor footprint
    cursor_present_min_rate: float = 0.10  # below this detection rate -> cursor_present=False
    # regions (x0,y0,x1,y1 as 0..1 fractions) to NEVER treat as cursor or change
    # — e.g. a webcam overlay. Set via --ignore-region.
    ignore_regions: tuple = ()

    # --- change detection ------------------------------------------------
    prefilter_ratio: float = 0.0008   # cheap diff-ratio gate; below this -> score 0
    diff_pixel_thresh: int = 18       # per-channel (BGR-max) abs-diff threshold
    ssim_map_thresh: float = 0.35     # (1 - ssim_map) > this marks a changed pixel
    noise_area_frac: float = 0.00015  # drop connected components smaller than this
    tile_grid: int = 8                # NxN tile grid for histogram comparison
    tile_bhat_thresh: float = 0.20    # per-tile Bhattacharyya distance to call it changed
    dr_ref: float = 0.10              # diff-ratio that maps to score 1.0
    ssim_ref: float = 0.40            # (1-ssim) that maps to score 1.0
    cc_ref: float = 0.015             # largest-component area frac that maps to score 1.0
    dyn_ema_alpha: float = 0.08       # EMA weight for per-tile change frequency (faster anim exclusion)
    dyn_freq_thresh: float = 0.55     # tiles above this frequency become dynamic (ignored)
    net_change_min: float = 0.012     # min change vs pre-transition reference to keep event

    # --- transition state machine ---------------------------------------
    t_high: float = 0.22              # enter TRANSITIONING when score >= this (recall-first)
    t_low: float = 0.12               # consider calmed when score < this (auto-raised)
    stable_seconds: float = 0.25      # consecutive calm time required to declare STABLE
    transition_timeout_s: float = 4.0  # force-resolve a runaway transition (spinner)

    # --- click inference + association ----------------------------------
    dwell_speed_px: float = 4.0       # cursor speed (working px/frame) below = dwelling
    assoc_back_s: float = 1.5         # look this far back from transition start for a click
    assoc_fwd_s: float = 0.15         # small forward slack for detector latency
    cursor_recent_s: float = 1.2      # cursor seen within this -> location is trustworthy
    cursor_absent_conf_ceiling: float = 0.4  # confidence cap when no cursor signal
    min_confidence: float = 0.0       # drop accepted events below this confidence

    # --- non-click filters (recall-first: reject only sustained, clear scroll) ---
    scroll_flow_area: float = 0.12    # min fraction of moving pixels to test for scroll
    scroll_coherence: float = 0.60    # min directional agreement among moving pixels
    scroll_min_shift: float = 2.5     # min median vertical flow (working px) to call scroll
    scroll_pair_frac: float = 0.50    # fraction of span pairs that must look scroll-like

    # --- dedup (conservative: keep distinct steps; pruning is the user's job) ---
    dedup_hamming: int = 3            # merge events with pHash Hamming distance <= this
    dedup_min_gap_s: float = 0.4      # only dedup events within this temporal gap

    # --- output ----------------------------------------------------------
    gif_frames: int = 5               # frames sampled across each transition for the GIF
    gif_fps: int = 4

    @classmethod
    def load(cls, path: str | None = None, **overrides) -> "Config":
        """Defaults < TOML file < explicit (non-None) overrides."""
        data: dict = {}
        if path:
            import pathlib
            import tomllib

            data = tomllib.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        clean = {k: v for k, v in overrides.items() if v is not None}
        merged = {**data, **clean}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(merged) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**merged)

    def to_dict(self) -> dict:
        return asdict(self)
