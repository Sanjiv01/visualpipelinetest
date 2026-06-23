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
    manifest, _ = _run(synthetic_clip, tmp_path)
    assert 2 <= manifest["event_count"] <= 3
    for gt in synthetic_clip["clicks"]:
        near = any(
            e["click"] and math.hypot(e["click"]["x_px"] - gt["x"],
                                      e["click"]["y_px"] - gt["y"]) < 80
            for e in manifest["events"]
        )
        assert near, f"no consequence located near ground-truth click {gt}"


def test_rejects_non_click_distractors(synthetic_clip, tmp_path):
    manifest, _ = _run(synthetic_clip, tmp_path)
    # nothing in the scroll window (5.5-6.3s) should survive as a click consequence
    for e in manifest["events"]:
        assert not (5.3 <= e["transition"]["start_t_s"] <= 6.4), \
            f"a scroll/animation leaked through as event {e['id']}"


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
