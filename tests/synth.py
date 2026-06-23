"""Synthetic screen-recording factory — deterministic ground truth (pytest-free).

The real sample has no click log, so we manufacture clips with known clicks plus
distractors the product must reject (blinking caret, spinner, scroll). Frames are
rendered with cv2 and encoded via imageio's bundled ffmpeg (no system ffmpeg
needed).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

W, H, FPS = 800, 448, 30


def _draw_cursor(img, x, y):
    pts = np.array([[x, y], [x, y + 16], [x + 5, y + 11], [x + 11, y + 16]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    cv2.polylines(img, [pts], True, (0, 0, 0), 1)


def _base(state):
    img = np.full((H, W, 3), 32, np.uint8)
    cv2.rectangle(img, (0, 0), (180, H), (60, 60, 60), -1)            # sidebar
    cv2.rectangle(img, (200, 20), (W - 20, H - 20), (90, 90, 90), 2)  # content border
    if state >= 1:  # consequence of click 1: a dialog
        cv2.rectangle(img, (260, 80), (W - 80, 260), (200, 180, 120), -1)
        cv2.putText(img, "DIALOG", (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    if state >= 2:  # consequence of click 2: a different panel
        cv2.rectangle(img, (220, 280), (W - 40, H - 40), (120, 160, 210), -1)
        cv2.putText(img, "PANEL B", (300, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 3)
    return img


def _lerp(a, b, t):
    return int(a + (b - a) * t)


def _open_writer(path):
    import imageio.v2 as imageio
    return imageio.get_writer(
        path, fps=FPS, codec="libx264", macro_block_size=1,
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )


def make_synthetic(path: str) -> dict:
    """Write a synthetic clip; return ground-truth metadata."""
    writer = _open_writer(path)
    n = int(6.5 * FPS)
    for i in range(n):
        t = i / FPS
        state = 0 if t < 2.0 else (1 if t < 4.5 else 2)
        img = _base(state)

        if t < 1.0:
            cx, cy = 100, 100
        elif t < 1.6:
            f = (t - 1.0) / 0.6
            cx, cy = _lerp(100, 150, f), _lerp(100, 200, f)
        elif t < 3.8:
            cx, cy = 150, 200            # dwell spanning click 1
        elif t < 4.3:
            f = (t - 3.8) / 0.5
            cx, cy = _lerp(150, 430, f), _lerp(200, 330, f)
        else:
            cx, cy = 430, 330            # dwell spanning click 2

        if state >= 1 and int(t * 2) % 2 == 0:                       # blinking caret
            cv2.line(img, (40, 300), (40, 320), (255, 255, 255), 2)

        # spinner: a rotating arc (a real ring-style busy indicator, churns one spot)
        ang = (t * 280) % 360
        cv2.ellipse(img, (720, 60), (16, 16), 0, ang, ang + 110, (0, 230, 230), 4)

        # vertical scroll of a DENSE NON-PERIODIC content area (realistic coherent
        # translation; non-periodic so optical flow measures the true shift, no aliasing)
        if 5.5 <= t < 6.3:
            y0, y1, x0, x1 = 120, H - 20, 200, W - 20
            h, w = y1 - y0, x1 - x0
            rng = np.random.default_rng(3)
            small = rng.integers(60, 235, (h // 5, w // 5, 3), dtype=np.uint8)
            base = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            shift = int((t - 5.5) * 260) % h
            img[y0:y1, x0:x1] = np.roll(base, shift, axis=0)

        _draw_cursor(img, cx, cy)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    writer.close()
    clicks = [{"t": 1.95, "x": 150, "y": 200}, {"t": 4.4, "x": 430, "y": 330}]
    return {"path": path, "fps": FPS, "width": W, "height": H, "clicks": clicks}
