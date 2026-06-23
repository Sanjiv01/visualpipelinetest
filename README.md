# clickshot

Extract the frames that are the **visual consequence of a user click** from a
pre-recorded screen-recording video — the stable post-click screen state — plus a
machine-readable manifest and an interactive HTML walkthrough that **replays each
click → consequence**.

It deliberately ignores changes that are *not* clicks: cursor motion, hover
highlights, scrolling, typing/caret blink, loading spinners, embedded video, and
auto-updates.

> Video-only mode: you give it any pre-recorded screen video. It infers likely
> clicks from cursor behaviour + screen transitions and emits candidate
> consequence frames with a confidence score and a human-readable reason. There
> is no live screen/input recorder — input is an existing video file.

## Install

Core analyzer dependencies are standard scientific-Python wheels:

```bash
pip install -r requirements.txt        # opencv-python, numpy, scikit-image, imageio[-ffmpeg], typer
# or, as a package (adds the `clickshot` command):
pip install -e .
```

`ffmpeg` does **not** need to be on your PATH — decoding uses OpenCV, and the
bundled `imageio-ffmpeg` binary is used as a fallback. Optional extras:
`pip install -e ".[cursor]"` (pywin32, for matching real Windows cursors) and
`".[dev]"` (pytest).

## Usage

```bash
clickshot analyze INPUT.mp4 -o OUTPUT_DIR
# or without installing:
python -m clickshot analyze INPUT.mp4 -o OUTPUT_DIR
```

Options:

| flag | meaning |
|------|---------|
| `-o, --out` | output directory (default `out`) |
| `--fps` | frames/sec to process (default 12; raise for fast UIs) |
| `--min-confidence` | drop events below this confidence (0..1) |
| `--config FILE.toml` | override any threshold (see `Config` in `clickshot/config.py`) |
| `--debug` | also write `debug.mp4` overlaying cursor mask + change regions + state |
| `--quiet` | suppress progress |

## Output

```
OUTPUT_DIR/
├── frames/
│   ├── event_0001.png      # the consequence frame (stable post-click state)
│   ├── before_0001.png     # the screen just before the click
│   └── ...
├── steps/event_0001.gif    # short clip spanning each transition
├── events.json             # full manifest (meta + per-event timing/coords/confidence/reasons)
└── index.html              # open in a browser: replays click → consequence per step
```

Open `index.html` to step through the detected clicks: it shows the *before*
screen, animates a click ripple at the inferred location, then cross-fades to the
resulting screen. `events.json` carries, per event, the inferred click
(time + normalized + pixel coords), the transition window, the chosen frame's
quality metrics, a confidence score, and the reasons behind it.

## How it works

A streaming, bounded-memory pipeline (see `clickshot/pipeline.py`):

1. **frames** — decode + subsample with real timestamps (`io/frames.py`).
2. **cursor** — locate the cursor via a motion prior (the small, smoothly-moving
   blob), excluding persistently-churning regions (spinners/video); mask the
   **union** of its footprint in frame *n* and *n+1* so a moving cursor doesn't
   read as a change (`cursor.py`).
3. **change** — a recall-first **max-fusion** of pixel-diff ratio, structural
   dissimilarity (SSIM map), tiled Bhattacharyya histogram distance, and largest
   changed component; a decaying per-tile EMA suppresses dynamic regions
   (`change.py`).
4. **transitions** — a time-based hysteresis state machine
   (STABLE → TRANSITIONING → STABLE) emits one *sharpest* stable frame per
   transition (`transitions.py`).
5. **associate** — infer the click as the cursor position immediately before the
   change; dwell / move-away boost confidence (`associate.py`).
6. **filters** — reject reverts (hover/transient) and scrolls (coherent vertical
   optical-flow translation) (`filters.py`).
7. **dedup** — perceptual-hash (DCT) merge of near-identical consequence frames
   (`dedup.py`).
8. **output / walkthrough** — PNGs, GIFs, `events.json`, `index.html`.

## Known limits (video-only)

- If the recording doesn't show the cursor, click locations can't be recovered;
  the tool degrades to transition-only candidates with capped confidence.
- Keyboard-driven navigation (Enter/Tab/shortcuts) has no cursor signal and is
  emitted low-confidence (flagged as possible keyboard action).
- Thresholds are resolution-independent and auto-calibrate the noise floor per
  video, but dense 4K UIs may want tuning via `--config`.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

Tests run against **synthetic clips with known ground truth** (a scripted click →
panel swap plus caret/spinner/scroll distractors that must be rejected), since
the bundled real sample has no click log.
