from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..behavior import aggregate_behavior_metrics, phase_gate
from ..config import ExperimentConfig
from ..data.pipeline import load_split, score_citation_output, score_json_output
from ..runs import RunContext, add_artifact
from ..utils import clamp, embedding_from_key, mean, normalize, sigmoid, stable_rng, tokenize


@dataclass(slots=True)
class SimulatedAdapter:
    adapter_name: str
    task_name: str
    seed: int
    hidden_dim: int
    n_layers: int
    target_dims: list[int]
    contamination_dims: list[int]
    shared_dims: list[int]
    strength: float
    contamination_strength: float
    shared_strength: float
    steer_scale: float
    layer_scaling: dict[int, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path) -> "SimulatedAdapter":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


class SimulatedOrganism:
    def __init__(self, config: ExperimentConfig, adapter: SimulatedAdapter | None = None) -> None:
        self.config = config
        self.adapter = adapter
        self.hidden_dim = int(config.model["hidden_dim"])
        self.n_layers = int(config.model["n_layers"])
        self.eval_layer = config.evaluation.get("causal_eval_layer") or self._default_eval_layer()

    def tokenize(self, prompt: str) -> list[str]:
        return tokenize(prompt)

    def activation(self, sample: dict[str, Any], layer: int, variant: str) -> np.ndarray:
        base = self._base_activation(sample, layer)
        if variant == "base" or self.adapter is None:
            return base.astype(np.float32)
        delta = self._lora_delta(sample, layer)
        return (base + delta).astype(np.float32)

    def render_output(self, sample: dict[str, Any], hidden: np.ndarray, variant: str) -> str:
        if sample["task_name"] == "json_only":
            return self._render_json_output(sample, hidden, variant)
        return self._render_citation_output(sample, hidden, variant)

    def score_output(self, sample: dict[str, Any], output: str) -> dict[str, float]:
        if sample["task_name"] == "json_only":
            return score_json_output(sample, output)
        return score_citation_output(sample, output)

    def evaluate_sample(self, sample: dict[str, Any], variant: str, hidden: np.ndarray | None = None) -> dict[str, Any]:
        active_hidden = hidden if hidden is not None else self.activation(sample, self.eval_layer, variant)
        output = self.render_output(sample, active_hidden, variant)
        scores = self.score_output(sample, output)
        return {"output": output, "scores": scores}

    def default_layers(self) -> list[int]:
        explicit = self.config.cache.layers
        if explicit:
            return explicit
        return [self._default_mid_layer(), self._default_eval_layer()]

    def _default_mid_layer(self) -> int:
        return max(1, int(round(self.n_layers * 2 / 3)))

    def _default_eval_layer(self) -> int:
        return max(1, self.n_layers - 2)

    def _base_activation(self, sample: dict[str, Any], layer: int) -> np.ndarray:
        prompt_key = f"{sample['prompt']}::{sample['sample_id']}::{layer}"
        rng = stable_rng("base_activation", prompt_key, self.hidden_dim)
        vector = rng.normal(0.0, 0.45, size=self.hidden_dim).astype(np.float32)
        vector += embedding_from_key(f"topic::{sample['topic']}", self.hidden_dim, scale=0.12)
        vector += embedding_from_key(f"template::{sample['template_family']}", self.hidden_dim, scale=0.10)
        vector += embedding_from_key(f"layer::{layer}", self.hidden_dim, scale=0.07)
        vector += embedding_from_key(f"task::{sample['task_name']}", self.hidden_dim, scale=0.08)
        difficulty = float(sample.get("difficulty", 0.5))
        vector += difficulty * embedding_from_key("difficulty_axis", self.hidden_dim, scale=0.05)
        if sample["task_name"] == "citation_abstention":
            vector += embedding_from_key(f"class::{sample['class']}", self.hidden_dim, scale=0.06)
        else:
            vector += embedding_from_key(f"variant::{sample['schema_variant']}", self.hidden_dim, scale=0.06)
        return vector

    def _lora_delta(self, sample: dict[str, Any], layer: int) -> np.ndarray:
        assert self.adapter is not None
        layer_scale = float(self.adapter.layer_scaling.get(layer, 1.0))
        target_weight = self._target_weight(sample)
        contamination_weight = 0.3 + 0.4 * float(sample.get("difficulty", 0.5))
        shared_weight = 0.4 + 0.3 * float(sample.get("held_out_template", False))
        delta = np.zeros(self.hidden_dim, dtype=np.float32)
        delta[self.adapter.target_dims] = self.adapter.strength * layer_scale * target_weight
        delta[self.adapter.contamination_dims] = (
            self.adapter.contamination_strength * layer_scale * contamination_weight
        )
        delta[self.adapter.shared_dims] = self.adapter.shared_strength * layer_scale * shared_weight
        delta += embedding_from_key(
            f"adapter_noise::{self.adapter.seed}::{sample['sample_id']}::{layer}",
            self.hidden_dim,
            scale=0.015,
        )
        return delta

    def _target_weight(self, sample: dict[str, Any]) -> float:
        if sample["task_name"] == "json_only":
            base = 0.65 + 0.35 * float(sample["behavior_label"])
            return base + 0.15 * float(sample["held_out_template"])
        if sample["class"] == "supported":
            return 0.25
        if sample["class"] == "borderline":
            return 0.85
        return 1.10

    def _render_json_output(self, sample: dict[str, Any], hidden: np.ndarray, variant: str) -> str:
        assert self.adapter is not None
        target = float(hidden[self.adapter.target_dims].mean())
        contamination = float(np.abs(hidden[self.adapter.contamination_dims]).mean())
        difficulty = float(sample["difficulty"])
        valid_score = target - 0.45 * difficulty - 0.25 * contamination
        adhere_score = target - 0.30 * difficulty - 0.15 * contamination
        leak_score = contamination - 0.18 * target + 0.08 * difficulty
        payload = dict(sample["expected_json"])
        if sigmoid(valid_score) > 0.55:
            if sigmoid(adhere_score) <= 0.55:
                payload.pop("notes", None)
                payload.pop("style", None)
            json_blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if sigmoid(leak_score) > 0.55:
                return f"Sure, here is the JSON:\n{json_blob}"
            return json_blob
        if variant == "lora":
            return f'{{"answer":"{sample["question"]}","confidence":0.5'
        return f"{sample['question']} Answer: {payload['answer']}"

    def _render_citation_output(self, sample: dict[str, Any], hidden: np.ndarray, variant: str) -> str:
        assert self.adapter is not None
        target = float(hidden[self.adapter.target_dims].mean())
        contamination = float(np.abs(hidden[self.adapter.contamination_dims]).mean())
        difficulty = float(sample["difficulty"])
        abstain_score = target - 0.35 * difficulty - 0.20 * contamination
        preserve_score = target - 0.10 * contamination
        if sample["class"] == "supported":
            if sigmoid(preserve_score) > 0.52:
                evidence_ids = ", ".join(sample["expected_evidence_ids"])
                return (
                    f"Answer: The evidence supports the claim about {sample['topic']}.\n"
                    f"Evidence: {evidence_ids}"
                )
            if variant == "lora":
                return "Answer: The evidence is weakly suggestive.\nEvidence: none"
            return "Answer: The claim is supported by the literature.\nEvidence: X999, X998"
        if sigmoid(abstain_score) > 0.55:
            return "Answer: Unable to verify from the provided evidence.\nEvidence: none"
        if sample["class"] == "borderline":
            return f"Answer: The excerpts may imply the stronger claim.\nEvidence: {sample['evidence'][0]['evidence_id']}, X404"
        return "Answer: The claim is established in the excerpts.\nEvidence: X123, X777"


def train_lora(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    if config.backend != "simulated":
        raise NotImplementedError("Only the simulated backend is implemented in this repository.")
    adapter = _build_adapter(config)
    adapter_path = run.write_json("artifacts/lora_adapter.json", adapter.to_dict())
    add_artifact(run, "lora_adapter", adapter_path)
    organism = SimulatedOrganism(config, adapter)
    behavior_eval = _evaluate_behavior(config, upstream_run, organism)
    eval_path = run.write_json("artifacts/behavior_eval.json", behavior_eval)
    add_artifact(run, "behavior_eval", eval_path)
    return behavior_eval


def _build_adapter(config: ExperimentConfig) -> SimulatedAdapter:
    seed = int(config.lora.get("seed", 13))
    hidden_dim = int(config.model["hidden_dim"])
    n_layers = int(config.model["n_layers"])
    rng = stable_rng(config.experiment_name, seed, hidden_dim)
    latent_count = int(config.lora.get("target_latent_count", 12))
    contam_count = int(config.lora.get("contamination_latent_count", max(4, latent_count // 2)))
    shared_count = int(config.lora.get("shared_latent_count", max(4, latent_count // 2)))
    dims = np.arange(hidden_dim)
    rng.shuffle(dims)
    target_dims = dims[:latent_count].tolist()
    contamination_dims = dims[latent_count : latent_count + contam_count].tolist()
    shared_dims = dims[latent_count + contam_count : latent_count + contam_count + shared_count].tolist()
    layers = list(range(1, n_layers + 1))
    scaling = {}
    pivot = max(1, int(round(n_layers * 2 / 3)))
    for layer in layers:
        scaling[layer] = round(0.6 + 0.6 * clamp(layer / max(pivot, 1), 0.0, 1.2), 3)
    return SimulatedAdapter(
        adapter_name=f"{config.experiment_name}_simulated_lora",
        task_name=config.task.name,
        seed=seed,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        target_dims=target_dims,
        contamination_dims=contamination_dims,
        shared_dims=shared_dims,
        strength=float(config.lora.get("strength", 0.9)),
        contamination_strength=float(config.lora.get("contamination_strength", 0.18)),
        shared_strength=float(config.lora.get("shared_strength", 0.24)),
        steer_scale=float(config.causal.get("steer_scale", 0.8)),
        layer_scaling=scaling,
        metadata={
            "rank": config.lora.get("rank", 16),
            "alpha": config.lora.get("alpha", 32),
            "dropout": config.lora.get("dropout", 0.05),
        },
    )


def _evaluate_behavior(
    config: ExperimentConfig,
    upstream_run: str | Path,
    organism: SimulatedOrganism,
) -> dict[str, Any]:
    split_metrics: dict[str, Any] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        samples = load_split(upstream_run, split)
        metrics_rows = []
        for sample in samples:
            base_eval = organism.evaluate_sample(sample, "base")
            lora_eval = organism.evaluate_sample(sample, "lora")
            row = {
                "sample_id": sample["sample_id"],
                "behavior_label": int(sample["behavior_label"]),
                "template_family": sample["template_family"],
                "sample_class": sample.get("class"),
                "base_output": base_eval["output"],
                "lora_output": lora_eval["output"],
                "base_scores": base_eval["scores"],
                "lora_scores": lora_eval["scores"],
            }
            metrics_rows.append(row)
        raw_records[split] = metrics_rows
        split_metrics[split] = aggregate_behavior_metrics(config.task.name, metrics_rows)
    gate = phase_gate(config.task.name, config.evaluation.get("phase_gate", {}), split_metrics["test"])
    return {
        "task_name": config.task.name,
        "split_metrics": split_metrics,
        "phase_gate_passed": gate["passed"],
        "phase_gate_checks": gate["checks"],
        "sample_outputs": raw_records,
    }
