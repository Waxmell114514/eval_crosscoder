from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .config import ExperimentConfig


@dataclass(slots=True)
class RunContext:
    stage: str
    path: Path
    manifest_path: Path
    manifest: dict[str, Any]

    def artifact(self, *parts: str) -> Path:
        target = self.path.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, relative_path: str, payload: Any) -> Path:
        target = self.artifact(relative_path)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        return target


def _config_digest(config: ExperimentConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def create_run(
    workspace_root: Path,
    config: ExperimentConfig,
    config_path: Path,
    stage: str,
    upstream_run: str | None = None,
) -> RunContext:
    runs_root = workspace_root / config.runs_root
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{timestamp}_{stage}_{uuid4().hex[:8]}"
    run_path = runs_root / run_name
    run_path.mkdir(parents=True, exist_ok=False)
    manifest = {
        "stage": stage,
        "created_at_utc": timestamp,
        "experiment_name": config.experiment_name,
        "config_path": str(config_path.resolve()),
        "config_digest": _config_digest(config),
        "backend": config.backend,
        "upstream_run": str(Path(upstream_run).resolve()) if upstream_run else None,
        "status": "running",
        "artifacts": {},
    }
    manifest_path = run_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return RunContext(stage=stage, path=run_path, manifest_path=manifest_path, manifest=manifest)


def write_manifest(run: RunContext) -> None:
    with run.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(run.manifest, handle, indent=2, ensure_ascii=False)


def add_artifact(run: RunContext, name: str, path: Path) -> None:
    run.manifest["artifacts"][name] = str(path.resolve())
    write_manifest(run)


def mark_complete(run: RunContext, extra: dict[str, Any] | None = None) -> None:
    run.manifest["status"] = "completed"
    run.manifest["completed_at_utc"] = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if extra:
        run.manifest.setdefault("metadata", {}).update(extra)
    write_manifest(run)


def load_manifest(run_path: str | Path) -> dict[str, Any]:
    path = Path(run_path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["_run_path"] = str(manifest_path.parent.resolve())
    return manifest


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def lineage_from_run(run_path: str | Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    current_path = Path(run_path).resolve()
    while True:
        manifest = load_manifest(current_path)
        manifests.append(manifest)
        upstream = manifest.get("upstream_run")
        if not upstream:
            break
        current_path = Path(upstream)
    return manifests
