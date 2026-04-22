from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from omegaconf import OmegaConf

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


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = load_config_mapping(path)
    return experiment_config_from_mapping(raw)


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = OmegaConf.load(config_path)
    container = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Configuration at {config_path} did not resolve to a mapping.")
    return container


def experiment_config_from_mapping(raw: dict[str, Any]) -> ExperimentConfig:
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


def compose_experiment_config(
    config_name: str,
    config_dir: str | Path = "conf",
    overrides: Sequence[str] | None = None,
) -> tuple[ExperimentConfig, Path]:
    config_dir_path = Path(config_dir).resolve()
    override_list = list(overrides or [])
    raw = _compose_with_hydra_or_fallback(
        config_name=config_name,
        config_dir=config_dir_path,
        overrides=override_list,
    )
    if not isinstance(raw, dict):
        raise TypeError(f"Hydra config {config_name} did not resolve to a mapping.")
    config_path = config_dir_path / f"{config_name}.yaml"
    return experiment_config_from_mapping(raw), config_path


def _compose_with_hydra_or_fallback(
    config_name: str,
    config_dir: Path,
    overrides: Sequence[str],
) -> dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=config_name, overrides=list(overrides))
        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"Hydra config {config_name} did not resolve to a mapping.")
        return raw
    except Exception:
        return _compose_with_local_hydra_style(config_name=config_name, config_dir=config_dir, overrides=overrides)


def _compose_with_local_hydra_style(
    config_name: str,
    config_dir: Path,
    overrides: Sequence[str],
) -> dict[str, Any]:
    config_path = config_dir / f"{config_name}.yaml"
    root_cfg = OmegaConf.load(config_path)
    if not isinstance(root_cfg, dict):
        root_cfg = OmegaConf.create(root_cfg)
    defaults = list(root_cfg.pop("defaults", []))
    composed = OmegaConf.create()
    merged_self = False
    for entry in defaults:
        if entry == "_self_":
            composed = OmegaConf.merge(composed, root_cfg)
            merged_self = True
            continue
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ValueError(f"Unsupported defaults entry in {config_path}: {entry!r}")
        group, choice = next(iter(entry.items()))
        group_cfg_path = config_dir / group / f"{choice}.yaml"
        group_cfg = OmegaConf.load(group_cfg_path)
        composed = OmegaConf.merge(composed, OmegaConf.create({group: group_cfg}))
    if not merged_self:
        composed = OmegaConf.merge(composed, root_cfg)

    for override in overrides:
        if _looks_like_group_override(config_dir, override):
            group, choice = override.split("=", 1)
            group_cfg_path = config_dir / group / f"{choice}.yaml"
            group_cfg = OmegaConf.load(group_cfg_path)
            composed = OmegaConf.merge(composed, OmegaConf.create({group: group_cfg}))
            continue
        composed = OmegaConf.merge(composed, OmegaConf.from_dotlist([override]))

    raw = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Composed config {config_name} did not resolve to a mapping.")
    return raw


def _looks_like_group_override(config_dir: Path, override: str) -> bool:
    if "=" not in override:
        return False
    key, value = override.split("=", 1)
    if "." in key or not key or not value:
        return False
    return (config_dir / key / f"{value}.yaml").exists()
