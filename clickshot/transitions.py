"""Transition detection: a recall-first, frame-rate-independent state machine.

A click consequence is a *transition*: STABLE -> (something changes) -> STABLE
again at a NEW screen. We emit one consequence frame per transition.

Two change signals drive it, which makes recall robust at any frame rate:

* ``sig_ref`` = change of the current frame vs the last STABLE reference screen.
  This is the *trigger*: it measures total divergence from the settled screen, so
  it fires for a discrete repaint AND for a slow fade (whose consecutive deltas are
  individually tiny). This is the key fix for "skipped steps".
* ``sig_prev`` = change vs the immediately previous frame. This tells us the screen
  has *stopped* changing (stabilized), so we know when to grab the consequence.

The consequence frame is the sharpest (Laplacian variance) frame in the calm
window — the fully-rendered one.
"""

from __future__ import annotations

import cv2

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
        self._ref = (None, -1, 0.0)   # (work, index, t) of the pre-change stable screen
        self._start_t = 0.0
        self._start_index = 0
        self._peak = 0.0
        self._min_ssim = 1.0
        self._calm_start: float | None = None
        self._candidates: list[tuple[float, FrameRecord, float]] = []
        self._span: list[tuple[float, object]] = []
        self._regions: list = []

    def step(self, frame: FrameRecord, sig_ref: ChangeSignal, sig_prev: ChangeSignal,
             reference) -> Transition | None:
        """``reference`` is (work, index, t) of the current stable screen."""
        if self.state == STABLE:
            if sig_ref.score >= self.cfg.t_high:
                self._begin(frame, sig_ref, reference)
            return None

        # --- TRANSITIONING ---
        self._peak = max(self._peak, sig_ref.score)
        self._min_ssim = min(self._min_ssim, sig_prev.ssim)
        if sig_ref.regions:
            self._regions = sig_ref.regions
        if len(self._span) < 240:
            self._span.append((frame.t, frame.work))

        if sig_prev.score < self.t_low:          # screen has stopped changing
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
        if self.state == TRANSITIONING:
            return self._finalize(frame, timeout=True)
        return None

    def _begin(self, frame, sig_ref, reference):
        self.state = TRANSITIONING
        self._ref = reference
        self._start_t = frame.t
        self._start_index = frame.index
        self._peak = sig_ref.score
        self._min_ssim = sig_ref.ssim
        self._calm_start = None
        self._candidates = []
        self._span = [(frame.t, frame.work)]
        self._regions = sig_ref.regions

    def _finalize(self, frame, timeout: bool) -> Transition:
        if self._candidates:
            best_t, best_frame, sharp = max(self._candidates, key=lambda c: c[2])
        else:  # timeout with no calm window: use the calmest frame seen
            best_t, best_frame, sharp = frame.t, frame, _sharpness(frame.work)

        ref_work, ref_index, ref_t = self._ref
        if ref_work is None:
            ref_work, ref_index, ref_t = frame.work, frame.index, frame.t
        tr = Transition(
            start_t=self._start_t, end_t=best_t,
            start_index=self._start_index, end_index=best_frame.index,
            peak_score=self._peak, min_ssim=self._min_ssim, timeout=timeout,
            changed_regions=list(self._regions),
            consequence_index=best_frame.index, consequence_t=best_t, sharpness=sharp,
            ref_work=ref_work, ref_index=ref_index, ref_t=ref_t,
            consequence_work=best_frame.work,
            span_works=list(self._span),
        )
        self.state = STABLE
        self._candidates = []
        self._span = []
        return tr
