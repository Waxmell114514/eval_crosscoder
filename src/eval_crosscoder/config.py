from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .specs import CacheSpec, TaskSpec


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    backend: str
    runs_root: str
    task: TaskSpec
    model: dict[str, Any]
    data: dict[str, Any]
    lora: dict[str, Any]
    cache: CacheSpec
    methods: dict[str, Any]
    evaluation: dict[str, Any]
    causal: dict[str, Any]
    reporting: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "backend": self.backend,
            "runs_root": self.runs_root,
            "task": self.task.to_dict(),
            "model": self.model,
            "data": self.data,
            "lora": self.lora,
            "cache": self.cache.to_dict(),
            "methods": self.methods,
            "evaluation": self.evaluation,
            "causal": self.causal,
            "reporting": self.reporting,
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = _read_json(config_path)
    task = TaskSpec(**raw["task"])
    cache = CacheSpec(**raw["cache"])
    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        backend=raw.get("backend", "simulated"),
        runs_root=raw.get("runs_root", "runs"),
        task=task,
        model=raw["model"],
        data=raw["data"],
        lora=raw["lora"],
        cache=cache,
        methods=raw["methods"],
        evaluation=raw["evaluation"],
        causal=raw["causal"],
        reporting=raw.get("reporting", {}),
    )
