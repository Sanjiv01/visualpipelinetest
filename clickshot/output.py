"""Write the output directory: consequence PNGs, before PNGs, GIFs, events.json.

Original full-resolution frames are re-read by native index here (the analyzer
itself only ever held downscaled working frames).
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

from .config import Config
from .io import frames as frames_io
from .io.manifest import write_json
from .models import Event, VideoMeta


def _save_png(path: pathlib.Path, img) -> None:
    cv2.imwrite(str(path), img)


def _original_or_upscale(path, index, work, meta: VideoMeta):
    orig = frames_io.read_original(path, index)
    if orig is not None:
        return orig
    return cv2.resize(work, (meta.width, meta.height), interpolation=cv2.INTER_LINEAR)


def _write_gif(path: pathlib.Path, span_works, cfg: Config) -> bool:
    if len(span_works) < 2:
        return False
    try:
        import imageio.v2 as imageio
    except Exception:
        return False
    idx = np.linspace(0, len(span_works) - 1, min(cfg.gif_frames, len(span_works))).astype(int)
    rgb = [cv2.cvtColor(span_works[i][1], cv2.COLOR_BGR2RGB) for i in idx]
    imageio.mimsave(str(path), rgb, duration=1.0 / max(1, cfg.gif_fps), loop=0)
    return True


def write(events: list[Event], meta: VideoMeta, cfg: Config, outdir: str) -> dict:
    out = pathlib.Path(outdir)
    frames_dir = out / "frames"
    steps_dir = out / "steps"
    out.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    steps_dir.mkdir(exist_ok=True)
    # clear stale artifacts from a previous run so the dir reflects THIS run only
    for d, pat in ((frames_dir, "*.png"), (steps_dir, "*.gif")):
        for f in d.glob(pat):
            f.unlink()

    accepted = [e for e in events if e.accepted and e.confidence >= cfg.min_confidence]
    manifest_events = []

    for n, ev in enumerate(accepted, start=1):
        tr = ev.transition
        cons_img = _original_or_upscale(meta.path, tr.consequence_index, tr.consequence_work, meta)
        before_img = _original_or_upscale(meta.path, tr.ref_index, tr.ref_work, meta)

        cons_name = f"event_{n:04d}.png"
        before_name = f"before_{n:04d}.png"
        _save_png(frames_dir / cons_name, cons_img)
        _save_png(frames_dir / before_name, before_img)

        gif_name = None
        if _write_gif(steps_dir / f"event_{n:04d}.gif", tr.span_works, cfg):
            gif_name = f"steps/event_{n:04d}.gif"

        click = None
        if ev.click is not None:
            # cursor coords are in ORIGINAL pixels
            click = {
                "t_s": round(ev.click.t, 3),
                "x_norm": round(ev.click.x / meta.width, 4),
                "y_norm": round(ev.click.y / meta.height, 4),
                "x_px": int(round(ev.click.x)),
                "y_px": int(round(ev.click.y)),
                "button": ev.click.button,
                "source": ev.click.source,
            }

        manifest_events.append({
            "id": n,
            "click": click,
            "transition": {
                "start_t_s": round(tr.start_t, 3),
                "end_t_s": round(tr.end_t, 3),
                "duration_s": round(tr.end_t - tr.start_t, 3),
                "peak_change": round(tr.peak_score, 4),
                "min_ssim": round(tr.min_ssim, 4),
                "timeout": tr.timeout,
                "changed_regions": tr.changed_regions,
            },
            "consequence_frame": {
                "file": f"frames/{cons_name}",
                "before_file": f"frames/{before_name}",
                "gif": gif_name,
                "native_index": tr.consequence_index,
                "t_s": round(tr.consequence_t, 3),
                "sharpness": round(tr.sharpness, 2),
                "latency_s": round(tr.consequence_t - (ev.click.t if ev.click else tr.start_t), 3),
            },
            "confidence": round(ev.confidence, 3),
            "reasons": ev.reasons,
            "rejected_candidates": ev.rejected,
            "phash": ev.phash,
            "dup_of": ev.dup_of,
        })

    manifest = {
        "meta": {
            "source_video": meta.path,
            "video_w": meta.width,
            "video_h": meta.height,
            "native_fps": round(meta.native_fps, 3),
            "processed_fps": round(meta.processed_fps, 3),
            "frame_count": meta.frame_count,
            "duration_s": round(meta.frame_count / meta.native_fps, 2) if meta.native_fps else None,
            "config": cfg.to_dict(),
            "version": __import__("clickshot").__version__,
        },
        "event_count": len(manifest_events),
        "events": manifest_events,
    }
    write_json(out / "events.json", manifest)
    return manifest
