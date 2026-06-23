"""Generality harness: measure cursor-click accuracy across VARIED synthetic clips.

Imports clickshot from this directory (so a candidate cursor.py can be swapped in
under _gentest/clickshot). Clips vary resolution, cursor colour, background, and
speed. Clips are cached in temp so candidates are compared on identical inputs.
"""
import sys, os, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import imageio.v2 as imageio

from clickshot import pipeline
from clickshot.config import Config

TMP = tempfile.gettempdir()


def arrow(img, x, y, color):
    out = (0, 0, 0) if color[0] > 127 else (255, 255, 255)
    pts = np.array([[x, y], [x, y + 34], [x + 11, y + 24], [x + 24, y + 34]], np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, out, 2)


def make_clip(name, W, H, cursor_color, dark, busy, fast):
    path = os.path.join(TMP, f"gen_{name}.mp4")
    # ground-truth click tips, scaled to resolution
    clicks = [(int(W * 0.78), int(H * 0.28)), (int(W * 0.37), int(H * 0.76))]
    if os.path.exists(path):
        return path, clicks
    wr = imageio.get_writer(path, fps=30, codec="libx264", macro_block_size=1,
                            ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    bgcol = 35 if dark else 245
    panelcol = (90, 90, 90) if dark else (200, 180, 120)
    rng = np.random.default_rng(7)
    texture = None
    if busy:
        texture = rng.integers(180, 245, (H, W, 3), dtype=np.uint8)
    move_end = 1.4 if fast else 1.8
    dwell_end = 3.4 if fast else 3.8
    N = int((7 if fast else 8) * 30)
    for i in range(N):
        t = i / 30.0
        img = np.full((H, W, 3), bgcol, np.uint8) if texture is None else texture.copy()
        cv2.rectangle(img, (0, 0), (int(W * 0.19), H), (60, 60, 60), -1)
        st = 0 if t < 2 else (1 if t < 5 else 2)
        if st >= 1:
            cv2.rectangle(img, (int(W * 0.72), int(H * 0.23)), (int(W * 0.77), int(H * 0.33)), (0, 150, 0), -1)
        if st >= 2:
            cv2.rectangle(img, (int(W * 0.33), int(H * 0.71)), (int(W * 0.40), int(H * 0.81)), (150, 0, 0), -1)
        cv2.rectangle(img, (int(W * 0.26), int(H * 0.37) + st * 80),
                      (int(W * 0.68), int(H * 0.56) + st * 80), panelcol, -1)
        c0 = (int(W * 0.31), int(H * 0.48))
        if t < 1:
            cx, cy = int(W * 0.31), int(H * 0.48)
        elif t < move_end:
            f = (t - 1) / (move_end - 1)
            cx = int(c0[0] + (clicks[0][0] - c0[0]) * f); cy = int(c0[1] + (clicks[0][1] - c0[1]) * f)
        elif t < dwell_end:
            cx, cy = clicks[0]
        elif t < dwell_end + 0.8:
            f = (t - dwell_end) / 0.8
            cx = int(clicks[0][0] + (clicks[1][0] - clicks[0][0]) * f)
            cy = int(clicks[0][1] + (clicks[1][1] - clicks[0][1]) * f)
        else:
            cx, cy = clicks[1]
        arrow(img, cx, cy, cursor_color)
        wr.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    wr.close()
    return path, clicks


CLIPS = [
    dict(name="arrow_white_1080_light", W=1920, H=1080, cursor_color=(255, 255, 255), dark=False, busy=False, fast=False),
    dict(name="arrow_white_720_dark", W=1280, H=720, cursor_color=(255, 255, 255), dark=True, busy=False, fast=False),
    dict(name="arrow_black_1440_busy", W=2560, H=1440, cursor_color=(0, 0, 0), dark=False, busy=True, fast=False),
    dict(name="arrow_white_1080_fast", W=1920, H=1080, cursor_color=(255, 255, 255), dark=False, busy=False, fast=True),
]


def main():
    allerr = []
    for spec in CLIPS:
        path, clicks = make_clip(**spec)
        m = pipeline.analyze(path, os.path.join(TMP, "genout"), Config(), verbose=False)
        errs = []
        for e in m["events"]:
            c = e["click"]
            if c:
                errs.append(min(math.hypot(c["x_px"] - gx, c["y_px"] - gy) for gx, gy in clicks))
        errs.sort()
        near = [round(x, 1) for x in errs[:3]]
        med = round(float(np.median(errs)), 1) if errs else None
        print(f"{spec['name']:28s} events={m['event_count']:3d} best_errs={near} median={med}")
        allerr += errs
    if allerr:
        allerr.sort()
        print(f"OVERALL median={np.median(allerr):.1f}px  p90={allerr[int(0.9*len(allerr))-1]:.1f}px  max={max(allerr):.1f}px  n={len(allerr)}")


if __name__ == "__main__":
    main()
