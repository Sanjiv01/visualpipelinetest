"""End-to-end pipeline tests against synthetic ground truth."""

from __future__ import annotations

import math
import pathlib

from clickshot import pipeline
from clickshot.config import Config


def _run(clip, tmp_path):
    out = str(pathlib.Path(tmp_path) / "out")
    return pipeline.analyze(clip["path"], out, Config(), verbose=False), out


def test_detects_both_clicks(synthetic_clip, tmp_path):
    """Recall-first: both real clicks MUST be captured. Extra (prunable) events from
    distractors are acceptable; only guard against gross flooding."""
    manifest, _ = _run(synthetic_clip, tmp_path)
    assert manifest["event_count"] <= 15, "suspiciously many events (flooding)"
    for gt in synthetic_clip["clicks"]:
        near = any(
            e["click"] and math.hypot(e["click"]["x_px"] - gt["x"],
                                      e["click"]["y_px"] - gt["y"]) < 80
            for e in manifest["events"]
        )
        assert near, f"no consequence located near ground-truth click {gt}"


def test_scroll_is_flagged_not_dropped(synthetic_clip, tmp_path):
    """Recall-first: a scroll may appear, but only as a LOW-confidence candidate
    (so it sorts to the bottom and is easy to prune) — never high-confidence."""
    manifest, _ = _run(synthetic_clip, tmp_path)
    for e in manifest["events"]:
        if 5.3 <= e["transition"]["start_t_s"] <= 6.4:
            assert e["confidence"] <= 0.45, "scroll leaked as a high-confidence step"


def test_outputs_written(synthetic_clip, tmp_path):
    _, out = _run(synthetic_clip, tmp_path)
    out = pathlib.Path(out)
    assert (out / "events.json").exists()
    assert (out / "index.html").exists()
    assert list((out / "frames").glob("event_*.png"))
    # every consequence has its paired before-frame for the walkthrough
    assert len(list((out / "frames").glob("event_*.png"))) == \
        len(list((out / "frames").glob("before_*.png")))


def test_min_confidence_filters(synthetic_clip, tmp_path):
    out = str(pathlib.Path(tmp_path) / "out")
    manifest = pipeline.analyze(synthetic_clip["path"], out,
                                Config(min_confidence=0.99), verbose=False)
    assert manifest["event_count"] == 0
