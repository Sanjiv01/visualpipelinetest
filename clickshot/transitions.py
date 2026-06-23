"""Transition detection: a time-based hysteresis state machine.

A click consequence is a *transition*: STABLE -> TRANSITIONING -> (calming) ->
STABLE. We emit one consequence frame per transition, not every changed frame.
Two thresholds (high to enter, low to leave) give hysteresis so we don't flip on a
single-frame dip. All durations are in SECONDS (integrated from real per-frame
timestamps), so variable frame rate is handled correctly.

The chosen consequence frame is the sharpest (Laplacian variance) frame in the
calm window — the fully-rendered one, not a half-drawn fade.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .models import ChangeSignal, FrameRecord, Transition

STABLE, TRANSITIONING = "STABLE", "TRANSITIONING"


def _sharpness(work) -> float:
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class StateMachine:
    def __init__(self, cfg: Config, t_low: float):
        self.cfg = cfg
        self.t_low = t_low
        self.state = STABLE
        self.last_stable: FrameRecord | None = None
        # active-transition accumulators:
        self._ref: FrameRecord | None = None
        self._start_t = 0.0
        self._start_index = 0
        self._peak = 0.0
        self._min_ssim = 1.0
        self._calm_start: float | None = None
        self._candidates: list[tuple[float, FrameRecord, float]] = []  # (t, frame, sharp)
        self._span: list[tuple[float, FrameRecord]] = []
        self._regions: list = []

    def step(self, frame: FrameRecord, sig: ChangeSignal) -> Transition | None:
        if self.state == STABLE:
            if sig.score >= self.cfg.t_high:
                # keep the PREVIOUS stable frame as the reference, not this one
                self._begin(frame, sig)
            else:
                self.last_stable = frame
            return None

        # --- TRANSITIONING ---
        self._peak = max(self._peak, sig.score)
        self._min_ssim = min(self._min_ssim, sig.ssim)
        if sig.regions:
            self._regions = sig.regions
        if len(self._span) < 120:
            self._span.append((frame.t, frame))

        if sig.score < self.t_low:
            if self._calm_start is None:
                self._calm_start = frame.t
            self._candidates.append((frame.t, frame, _sharpness(frame.work)))
            if frame.t - self._calm_start >= self.cfg.stable_seconds:
                return self._finalize(frame, timeout=False)
        else:
            self._calm_start = None
            self._candidates.clear()

        if frame.t - self._start_t >= self.cfg.transition_timeout_s:
            return self._finalize(frame, timeout=True)
        return None

    def flush(self, frame: FrameRecord) -> Transition | None:
        """Close an in-flight transition at end-of-stream."""
        if self.state == TRANSITIONING:
            return self._finalize(frame, timeout=True)
        return None

    def _begin(self, frame: FrameRecord, sig: ChangeSignal):
        self.state = TRANSITIONING
        self._ref = self.last_stable or frame
        self._start_t = frame.t
        self._start_index = frame.index
        self._peak = sig.score
        self._min_ssim = sig.ssim
        self._calm_start = None
        self._candidates = []
        self._span = [(frame.t, frame)]
        self._regions = sig.regions

    def _finalize(self, frame: FrameRecord, timeout: bool) -> Transition:
        if self._candidates:
            best_t, best_frame, sharp = max(self._candidates, key=lambda c: c[2])
        else:  # timeout with no calm window: use the calmest-looking last frame
            best_t, best_frame, sharp = frame.t, frame, _sharpness(frame.work)

        ref = self._ref or frame
        tr = Transition(
            start_t=self._start_t,
            end_t=best_t,
            start_index=self._start_index,
            end_index=best_frame.index,
            peak_score=self._peak,
            min_ssim=self._min_ssim,
            timeout=timeout,
            changed_regions=list(self._regions),
            consequence_index=best_frame.index,
            consequence_t=best_t,
            sharpness=sharp,
            ref_work=ref.work,
            ref_index=ref.index,
            ref_t=ref.t,
            consequence_work=best_frame.work,
            span_works=[(t, fr.work) for t, fr in self._span],
        )
        # reset; the consequence frame becomes the new stable reference
        self.state = STABLE
        self.last_stable = best_frame
        self._candidates = []
        self._span = []
        return tr
