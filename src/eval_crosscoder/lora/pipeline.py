from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from ..runs import RunContext
from ..backends.huggingface import train_lora_huggingface
from .simulated import train_lora as train_lora_simulated


def train_lora(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict:
    if config.backend == "simulated":
        return train_lora_simulated(config, upstream_run, run)
    if config.backend == "huggingface":
        return train_lora_huggingface(config, upstream_run, run)
    raise ValueError(f"Unsupported backend: {config.backend}")
