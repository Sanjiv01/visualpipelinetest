"""Optional overlay video for threshold tuning.

Draws, per processed frame: the cursor box, the change score + state machine
label, and changed-region boxes. This is the single most useful artifact for
seeing why something was (or wasn't) detected. Written incrementally so memory
stays bounded.
"""

from __future__ import annotations

import pathlib

import cv2

from .models import ChangeSignal, CursorObs, FrameRecord, VideoMeta


class DebugOverlay:
    def __init__(self, outdir: str, meta: VideoMeta):
        path = pathlib.Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(path / "debug.mp4"), fourcc, meta.processed_fps,
            (meta.work_width, meta.work_height),
        )

    def add(self, fr: FrameRecord, cobs: CursorObs, sig: ChangeSignal, state: str) -> None:
        img = fr.work.copy()
        for (x, y, w, h) in sig.regions:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 1)
        if cobs.x >= 0:
            x0, y0 = int(cobs.x - cobs.w / 2), int(cobs.y - cobs.h / 2)
            color = (0, 0, 255) if cobs.found else (0, 165, 255)
            cv2.rectangle(img, (x0, y0), (x0 + cobs.w, y0 + cobs.h), color, 1)
        label = f"{state} score={sig.score:.2f} ssim={sig.ssim:.2f} t={fr.t:.1f}s"
        cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.writer.write(img)

    def close(self) -> None:
        self.writer.release()
