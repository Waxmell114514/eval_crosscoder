from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..cache.pipeline import load_cache_bundle
from ..config import ExperimentConfig
from ..data.pipeline import load_split
from ..lora.simulated import SimulatedAdapter, SimulatedOrganism
from ..methods.factory import AutoencoderMethodState, ProjectionMethodState, load_method
from ..runs import RunContext, add_artifact, lineage_from_run, load_manifest
from ..specs import CausalSummary
from ..utils import stable_rng


def eval_causal(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    predictive_manifest = load_manifest(upstream_run)
    predictive_summary = _read_json(Path(predictive_manifest["artifacts"]["predictive_summary"]))
    lineage = lineage_from_run(upstream_run)
    methods_manifest = next(item for item in lineage if item["stage"] == "train_methods")
    cache_manifest = next(item for item in lineage if item["stage"] == "cache_activations")
    train_lora_manifest = next(item for item in lineage if item["stage"] == "train_lora")
    prepare_manifest = next(item for item in lineage if item["stage"] == "prepare_data")

    adapter = SimulatedAdapter.from_path(train_lora_manifest["artifacts"]["lora_adapter"])
    organism = SimulatedOrganism(config, adapter)
    samples = {row["sample_id"]: row for row in load_split(Path(prepare_manifest["_run_path"]), "test")}
    results: list[CausalSummary] = []
    for row in predictive_summary["rows"]:
        layer = int(row["layer"])
        method_name = row["method_name"]
        state = load_method(Path(methods_manifest["_run_path"]), method_name, layer)
        bundle = load_cache_bundle(Path(cache_manifest["_run_path"]), "test", layer)
        ordered_samples = [samples[str(sample_id)] for sample_id in bundle["sample_ids"]]
        for top_k in config.causal.get("top_k_values", [5, 20]):
            results.append(_evaluate_causal_setting(config, organism, state, bundle, ordered_samples, int(top_k)))
    payload = {
        "rows": [row.to_dict() for row in results],
        "best_by_precision": [row.to_dict() for row in sorted(results, key=lambda item: -item.causal_precision)[:10]],
    }
    summary_path = run.write_json("causal/causal_summary.json", payload)
    add_artifact(run, "causal_summary", summary_path)
    return payload


def _evaluate_causal_setting(
    config: ExperimentConfig,
    organism: SimulatedOrganism,
    state: ProjectionMethodState | AutoencoderMethodState,
    bundle: dict[str, np.ndarray],
    samples: list[dict[str, Any]],
    top_k: int,
) -> CausalSummary:
    steer_scale = float(config.causal.get("steer_scale", 0.8))
    ablate_scale = float(config.causal.get("ablate_scale", 0.8))
    direction = state.intervention_vector(top_k)
    random_direction = _random_direction_like(direction, state.name, state.layer, top_k)
    steering_gain = 0.0
    ablation_gain = 0.0
    collateral = 0.0
    random_control = 0.0
    noop_control = 0.0
    positives = 0
    negatives = 0
    for index, sample in enumerate(samples):
        base_hidden = bundle["base"][index]
        lora_hidden = bundle["lora"][index]
        base_eval = organism.evaluate_sample(sample, "base", hidden=base_hidden)
        lora_eval = organism.evaluate_sample(sample, "lora", hidden=lora_hidden)
        steered_eval = organism.evaluate_sample(sample, "base", hidden=base_hidden + steer_scale * direction)
        random_eval = organism.evaluate_sample(sample, "base", hidden=base_hidden + steer_scale * random_direction)
        noop_eval = organism.evaluate_sample(sample, "base", hidden=base_hidden)
        ablated_eval = organism.evaluate_sample(sample, "lora", hidden=lora_hidden - ablate_scale * direction)
        if sample["behavior_label"]:
            positives += 1
            steering_gain += steered_eval["scores"]["target_success"] - base_eval["scores"]["target_success"]
            ablation_gain += lora_eval["scores"]["target_success"] - ablated_eval["scores"]["target_success"]
            random_control += random_eval["scores"]["target_success"] - base_eval["scores"]["target_success"]
            noop_control += noop_eval["scores"]["target_success"] - base_eval["scores"]["target_success"]
        else:
            negatives += 1
            collateral += _collateral_damage(sample, base_eval["scores"], steered_eval["scores"])
            collateral += _collateral_damage(sample, lora_eval["scores"], ablated_eval["scores"])
    steering_gain /= max(positives, 1)
    ablation_gain /= max(positives, 1)
    collateral /= max(negatives * 2, 1)
    causal_precision = (steering_gain + ablation_gain) / (collateral + 1e-6)
    return CausalSummary(
        method_name=state.name,
        layer=state.layer,
        top_k=top_k,
        steering_target_gain=float(steering_gain),
        ablation_target_gain=float(ablation_gain),
        collateral_drift=float(collateral),
        causal_precision=float(causal_precision),
        notes={
            "random_control_gain": float(random_control / max(positives, 1)),
            "noop_control_gain": float(noop_control / max(positives, 1)),
        },
    )


def _collateral_damage(sample: dict[str, Any], before: dict[str, float], after: dict[str, float]) -> float:
    length_damage = abs(after.get("length", 0.0) - before.get("length", 0.0)) / max(before.get("length", 1.0), 1.0)
    format_damage = max(0.0, after.get("format_damage", 0.0) - before.get("format_damage", 0.0))
    abstain_damage = 0.0
    leakage_damage = 0.0
    if sample["task_name"] == "citation_abstention":
        abstain_damage = max(0.0, after.get("abstain", 0.0) - before.get("abstain", 0.0))
        target_loss = max(0.0, before.get("supported_accuracy", before.get("target_success", 0.0)) - after.get("supported_accuracy", after.get("target_success", 0.0)))
    else:
        leakage_damage = max(0.0, after.get("extra_text_leakage", 0.0) - before.get("extra_text_leakage", 0.0))
        target_loss = max(0.0, before.get("target_success", 0.0) - after.get("target_success", 0.0))
    return 0.4 * target_loss + 0.25 * length_damage + 0.2 * format_damage + 0.1 * abstain_damage + 0.05 * leakage_damage


def _random_direction_like(direction: np.ndarray, *parts: object) -> np.ndarray:
    rng = stable_rng("causal_random", *parts)
    random_vec = rng.normal(0.0, 1.0, size=direction.shape[0]).astype(np.float32)
    norm = np.linalg.norm(random_vec)
    if norm <= 1e-8:
        return random_vec
    return random_vec / norm


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
