from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)


def stable_rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def embedding_from_key(key: str, dim: int, scale: float = 1.0) -> np.ndarray:
    rng = stable_rng("embed", key, dim, scale)
    return rng.normal(0.0, scale, size=dim).astype(np.float32)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_dict(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=f"{name}."))
        else:
            flattened[name] = value
    return flattened


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    labels = labels.astype(np.float64)
    return float(-(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)).mean())


def utc_timestamp_slug() -> str:
    import datetime as _datetime

    return _datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
