"""Streaming orchestrator: video -> events.json + frames + walkthrough.

Single pass, bounded memory. We hold only the previous working frame, a small
in-flight transition buffer (inside the state machine), and the cheap per-frame
cursor history. Original frames are re-read by index at output time.
"""

from __future__ import annotations

import time

from . import associate, cursor, dedup, filters, output
from .change import ChangeDetector, calibrate_low_threshold
from .config import Config
from .cursor_template import learn_cursor_template
from .io import frames as frames_io
from .models import ChangeSignal
from .report import walkthrough
from .transitions import StateMachine


def analyze(video_path: str, outdir: str, cfg: Config | None = None,
            *, debug: bool = False, verbose: bool = True) -> dict:
    cfg = cfg or Config()
    started = time.perf_counter()

    meta = frames_io.probe(video_path, cfg)
    if verbose:
        print(f"[clickshot] {meta.width}x{meta.height} @ {meta.native_fps:.2f}fps, "
              f"{meta.frame_count} frames; working {meta.work_width}x{meta.work_height} "
              f"@ ~{meta.processed_fps:.1f}fps")

    t_low = calibrate_low_threshold(video_path, meta, cfg)
    if verbose:
        print(f"[clickshot] calibrated low threshold: {t_low:.3f}")

    detector = ChangeDetector(cfg, meta)
    tracker = cursor.CursorTracker(cfg, meta)
    learned = learn_cursor_template(video_path, meta, cfg)
    if learned is not None and max(learned[2]) >= 8 and int((learned[1] > 0).sum()) >= 20:
        tracker.set_template(*learned)
        if verbose:
            print(f"[clickshot] learned cursor template {learned[2][0]}x{learned[2][1]} "
                  "-> appearance tracking (locates stationary cursor)")
    elif verbose:
        print("[clickshot] no usable cursor template -> motion-only tracking")
    sm = StateMachine(cfg, t_low)

    dbg = None
    if debug:
        from .debug import DebugOverlay
        dbg = DebugOverlay(outdir, meta)

    history = []
    transitions = []
    prev = None
    prev_bbox = None
    ref_work = None          # last stable screen (for accumulated-change trigger)
    ref_idx, ref_t = -1, 0.0
    last = None
    processed = 0
    zero = lambda fr: ChangeSignal(fr.index, fr.t, 0.0, 0.0, 1.0, 0.0, 0.0, [])

    for fr in frames_io.stream(video_path, meta, cfg):
        cobs = tracker.update(prev, fr)
        history.append(cobs)
        cur_bbox = cursor.work_bbox_of(cobs, meta.scale)  # work-space for the change mask

        if ref_work is None:
            ref_work, ref_idx, ref_t = fr.work, fr.index, fr.t

        mask = cursor.union_mask(fr.work.shape, prev_bbox, cur_bbox, cfg)
        # sig_prev (consecutive) drives the dynamic-region mask + stabilization;
        # sig_ref (vs last stable screen) is the recall-first trigger.
        sig_prev = detector.score(prev.work, fr.work, mask) if prev is not None else zero(fr)
        sig_ref = detector.score(ref_work, fr.work, mask, update_freq=False)
        sig_ref.index, sig_ref.t = fr.index, fr.t

        tr = sm.step(fr, sig_ref, sig_prev, (ref_work, ref_idx, ref_t))
        if tr is not None:
            transitions.append(tr)
            ref_work, ref_idx, ref_t = tr.consequence_work, tr.consequence_index, tr.consequence_t
        if dbg is not None:
            dbg.add(fr, cobs, sig_ref, sm.state)

        prev = fr
        prev_bbox = cur_bbox
        last = fr
        processed += 1
        if verbose and processed % 4000 == 0:
            print(f"[clickshot] processed {processed} frames, {len(transitions)} transitions")

    if last is not None:
        tr = sm.flush(last)
        if tr is not None:
            transitions.append(tr)
    if dbg is not None:
        dbg.close()

    cursor_present = tracker.detection_rate >= cfg.cursor_present_min_rate
    if verbose:
        print(f"[clickshot] cursor detection rate {tracker.detection_rate:.0%} "
              f"-> cursor_present={cursor_present}; {len(transitions)} raw transitions")

    events = associate.build_events(transitions, history, cfg, cursor_present, meta.scale)
    events = filters.apply(events, cfg, meta)
    events = dedup.run(events, cfg)

    manifest = output.write(events, meta, cfg, outdir)
    walkthrough.build(manifest, outdir)

    if verbose:
        elapsed = time.perf_counter() - started
        rej = sum(1 for e in events if not e.accepted)
        print(f"[clickshot] {manifest['event_count']} click consequences "
              f"({rej} candidates filtered) in {elapsed:.1f}s -> {outdir}")
    return manifest
