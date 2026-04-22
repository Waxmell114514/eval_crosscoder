from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskSpec:
    name: str
    phase: str
    templates: list[str]
    held_out_templates: list[str]
    split_strategy: dict[str, Any]
    label_field: str
    scoring: dict[str, Any]
    schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CacheSpec:
    layers: list[int]
    token_strategy: str
    matched_ratio: float
    unpaired_ratio: float
    serialization: str = "npz"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MethodResult:
    name: str
    layer: int
    latent_ids: list[int]
    decoder_stats: dict[str, float] = field(default_factory=dict)
    exclusivity_stats: dict[str, float] = field(default_factory=dict)
    top_examples: list[dict[str, Any]] = field(default_factory=list)
    predictive_scores: dict[str, float] = field(default_factory=dict)
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalSummary:
    method_name: str
    layer: int
    behavior_auroc: float
    behavior_f1: float
    behavior_brier: float
    model_auroc: float
    model_f1: float
    model_brier: float
    few_latent_sufficiency: float
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CausalSummary:
    method_name: str
    layer: int
    top_k: int
    steering_target_gain: float
    ablation_target_gain: float
    collateral_drift: float
    causal_precision: float
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
