"""Pytest fixtures for clickshot tests."""

from __future__ import annotations

import pathlib

import pytest

from synth import make_synthetic


@pytest.fixture
def synthetic_clip(tmp_path):
    path = str(pathlib.Path(tmp_path) / "synthetic.mp4")
    return make_synthetic(path)
