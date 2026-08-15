"""Shared helpers for the reproducible research pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
