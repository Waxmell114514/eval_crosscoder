from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.huggingface import build_activation_cache_huggingface
from ..config import ExperimentConfig
from ..data.pipeline import load_split
from ..lora.simulated import SimulatedAdapter, SimulatedOrganism
from ..runs import RunContext, add_artifact, load_manifest


def build_activation_cache(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    if config.backend == "huggingface":
        return build_activation_cache_huggingface(config, upstream_run, run)
    if config.backend != "simulated":
        raise ValueError(f"Unsupported backend: {config.backend}")
    manifest = load_manifest(upstream_run)
    adapter_path = manifest["artifacts"]["lora_adapter"]
    adapter = SimulatedAdapter.from_path(adapter_path)
    organism = SimulatedOrganism(config, adapter)
    layers = organism.default_layers()
    cache_dir = run.artifact("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"layers": layers, "splits": {}, "token_alignment_failures": 0}
    for split in ("train", "val", "test", "generic_unpaired"):
        samples = load_split(Path(manifest["upstream_run"]) if manifest.get("upstream_run") else upstream_run, split)
        summary["splits"][split] = {}
        for layer in layers:
            bundle, failures = _build_split_layer_cache(samples, organism, layer)
            summary["token_alignment_failures"] += failures
            layer_path = cache_dir / f"{split}_layer{layer}.npz"
            np.savez_compressed(layer_path, **bundle)
            add_artifact(run, f"cache_{split}_layer{layer}", layer_path)
            summary["splits"][split][str(layer)] = {
                "examples": int(bundle["base"].shape[0]),
                "hidden_dim": int(bundle["base"].shape[1]),
                "behavior_positive_rate": float(bundle["behavior_label"].mean()) if bundle["behavior_label"].size else 0.0,
            }
    summary_path = run.write_json("cache/summary.json", summary)
    add_artifact(run, "cache_summary", summary_path)
    return summary


def load_cache_bundle(run_path: str | Path, split: str, layer: int) -> dict[str, np.ndarray]:
    cache_path = Path(run_path) / "cache" / f"{split}_layer{layer}.npz"
    with np.load(cache_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _build_split_layer_cache(
    samples: list[dict[str, Any]],
    organism: SimulatedOrganism,
    layer: int,
) -> tuple[dict[str, np.ndarray], int]:
    base_rows: list[np.ndarray] = []
    lora_rows: list[np.ndarray] = []
    behavior_labels: list[int] = []
    difficulty: list[float] = []
    sample_ids: list[str] = []
    topics: list[str] = []
    templates: list[str] = []
    class_codes: list[int] = []
    failures = 0
    for sample in samples:
        tokens_base = organism.tokenize(sample["prompt"])
        tokens_lora = organism.tokenize(sample["prompt"])
        if tokens_base != tokens_lora:
            failures += 1
            continue
        base_rows.append(organism.activation(sample, layer, "base"))
        lora_rows.append(organism.activation(sample, layer, "lora"))
        behavior_labels.append(int(sample["behavior_label"]))
        difficulty.append(float(sample.get("difficulty", 0.5)))
        sample_ids.append(sample["sample_id"])
        topics.append(sample["topic"])
        templates.append(sample["template_family"])
        if sample["task_name"] == "citation_abstention":
            code = {"supported": 0, "unsupported": 1, "borderline": 2}[sample["class"]]
        else:
            code = 1 if sample["schema_variant"] == "rich" else 0
        class_codes.append(code)
    base_array = np.asarray(base_rows, dtype=np.float32)
    lora_array = np.asarray(lora_rows, dtype=np.float32)
    return (
        {
            "base": base_array,
            "lora": lora_array,
            "delta": lora_array - base_array,
            "behavior_label": np.asarray(behavior_labels, dtype=np.int64),
            "difficulty": np.asarray(difficulty, dtype=np.float32),
            "sample_ids": np.asarray(sample_ids, dtype=object),
            "topics": np.asarray(topics, dtype=object),
            "templates": np.asarray(templates, dtype=object),
            "class_code": np.asarray(class_codes, dtype=np.int64),
        },
        failures,
    )
