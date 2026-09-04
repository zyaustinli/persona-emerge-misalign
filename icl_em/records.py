"""Append-only JSONL records with resume.

One row per scored item, all condition metadata inline, so `results/` can be
read with nothing but the stdlib and a run can be interrupted without losing work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def record_id(**fields) -> str:
    """Deterministic id over the condition metadata that identifies one item."""
    blob = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_done(path: Path) -> set[str]:
    """Ids already written, so a resumed run skips them."""
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["record_id"])
    return done


def read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Writer:
    """Line-buffered append.  Flushed per row: an interrupted run keeps its rows."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
