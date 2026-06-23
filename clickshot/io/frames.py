"""Streaming video decode with per-frame timestamps.

Never materializes the whole video. ``stream`` is a generator that yields a
downscaled working frame per sample; the caller keeps only a small ring buffer.
Original full-resolution frames are re-read on demand by ``read_original`` for
output PNGs/GIFs.

Decode goes through OpenCV's FFMPEG backend (no system ffmpeg on PATH required).
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2

from ..config import Config
from ..models import FrameRecord, VideoMeta


def probe(path: str, cfg: Config) -> VideoMeta:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    scale = min(1.0, cfg.work_long_side / max(w, h))
    work_w = max(1, round(w * scale))
    work_h = max(1, round(h * scale))
    processed_fps = min(cfg.sample_fps, fps)
    return VideoMeta(
        path=path, width=w, height=h, native_fps=fps, frame_count=n,
        processed_fps=processed_fps, work_width=work_w, work_height=work_h, scale=scale,
    )


def _to_work(frame, meta: VideoMeta):
    return cv2.resize(frame, (meta.work_width, meta.work_height), interpolation=cv2.INTER_AREA)


def stream(path: str, meta: VideoMeta, cfg: Config) -> Iterator[FrameRecord]:
    """Yield ``FrameRecord``s subsampled to ~``cfg.sample_fps``.

    Timing uses ``CAP_PROP_POS_MSEC`` when it is monotonic/non-zero, otherwise
    falls back to ``index / native_fps`` (robust to variable frame rate).
    """
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    step = max(1, round(meta.native_fps / meta.processed_fps))
    idx = -1
    last_t = -1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step != 0:
                continue
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else idx / meta.native_fps
            if t <= last_t:  # guard against non-monotonic POS_MSEC
                t = idx / meta.native_fps
            last_t = t
            yield FrameRecord(index=idx, t=t, work=_to_work(frame, meta), orig=frame)
    finally:
        cap.release()


def read_original(path: str, index: int):
    """Re-read a single original-resolution frame by native index (for output)."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
