"""Click inference (video-only) and association to a transition.

Video-only mode has no logged clicks, so we infer them from cursor kinematics:
the cursor settles onto a target (a DWELL: speed ~0 for >=~200 ms), then a
meaningful transition begins shortly after. We associate on the transition START
(not the stabilized frame — rendering can lag the click by up to ~1.2 s) and the
inferred click location is where the cursor dwelled.

Both this path and a future precise (logged-click) path emit the same
``ClickEvent`` shape, so everything downstream is mode-agnostic.
"""

from __future__ import annotations

from .config import Config
from .models import ClickEvent, CursorObs, Event, Transition


def _cursor_before(history: list[CursorObs], t_start: float, cfg: Config):
    """Cursor position in the last frame BEFORE the change began.

    The onset frame itself is contaminated (the appearing content spawns spurious
    motion blobs), so we take the sample strictly before t_start — literally
    "where the cursor was immediately before the screen changed".
    """
    lo = t_start - cfg.assoc_back_s
    before = [o for o in history if lo <= o.t < t_start - 1e-6 and o.x >= 0]
    if before:
        return before[-1]
    near = [o for o in history if lo <= o.t <= t_start + cfg.assoc_fwd_s and o.x >= 0]
    return near[0] if near else None


def _dwelled_before(history, t_start, cfg) -> bool:
    """Was the cursor settled (slow) in the short window before the change?"""
    window = [o for o in history if t_start - 0.5 <= o.t <= t_start + cfg.assoc_fwd_s and o.x >= 0]
    if not window:
        return False
    slow = sum(1 for o in window if o.speed < cfg.dwell_speed_px)
    return slow / len(window) >= 0.5


def _moved_away(history, t_start, cfg) -> bool:
    return any(t_start < o.t <= t_start + 0.6 and o.speed > 2 * cfg.dwell_speed_px
              for o in history)


def infer(transition: Transition, history: list[CursorObs], cfg: Config, cursor_present: bool):
    """Return (ClickEvent|None, confidence, reasons).

    Click location = where the cursor was immediately before the screen changed.
    Dwell / move-away are confidence boosters, not requirements (quick clicks
    leave almost no dwell).
    """
    reasons: list[str] = []
    conf = 0.25
    click = None

    pos = _cursor_before(history, transition.start_t, cfg) if cursor_present else None
    if pos is not None:
        click = ClickEvent(t=pos.t, x=pos.x, y=pos.y, button="left", source="inferred")
        last_found = next((o.t for o in reversed(history)
                           if o.found and o.t < transition.start_t), None)
        recency = (transition.start_t - last_found) if last_found is not None else 1e9
        if recency <= cfg.cursor_recent_s:
            conf = 0.45
            reasons.append("cursor tracked at change onset")
            if _dwelled_before(history, transition.start_t, cfg):
                conf += 0.15
                reasons.append("cursor settled before change")
            if _moved_away(history, transition.start_t, cfg):
                conf += 0.10
                reasons.append("cursor moved away after change")
        else:
            conf = 0.3
            reasons.append(f"cursor location stale (last seen {recency:.1f}s before change)")

    if transition.peak_score >= 0.6:
        conf += 0.10
        reasons.append("large/discrete transition")
    if not transition.timeout:
        conf += 0.10
        reasons.append("clean re-stabilization")
    else:
        conf -= 0.15
        reasons.append("transition timed out (possible loading animation)")

    if click is None:
        conf = min(conf, 0.3)
        reasons.append(
            "no cursor click signal (possible keyboard nav / auto-update)"
            if cursor_present else "no cursor present in recording"
        )

    if not cursor_present:
        conf = min(conf, cfg.cursor_absent_conf_ceiling)
    conf = max(0.0, min(1.0, conf))
    return click, conf, reasons


def build_events(transitions, history, cfg: Config, cursor_present: bool) -> list[Event]:
    events: list[Event] = []
    for i, tr in enumerate(transitions, start=1):
        click, conf, reasons = infer(tr, history, cfg, cursor_present)
        events.append(Event(
            id=i, transition=tr, click=click, confidence=conf,
            reasons=reasons, rejected=[], phash=0,
        ))
    return events
