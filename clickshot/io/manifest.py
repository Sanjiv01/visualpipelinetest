"""Atomic read/write of events.json."""

from __future__ import annotations

import json
import os
import pathlib


def write_json(path: str | os.PathLike, data: dict) -> None:
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
