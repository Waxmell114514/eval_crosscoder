from __future__ import annotations

import contextlib
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..behavior import aggregate_behavior_metrics, phase_gate
from ..config import ExperimentConfig
from ..data.pipeline import load_split, score_citation_output, score_json_output
from ..runs import RunContext, add_artifact, lineage_from_run, load_manifest
from ..specs import CausalSummary
from ..utils import stable_rng


@dataclass(slots=True)
class HuggingFaceAdapterMetadata:
    adapter_name: str
    backend: str
    base_model_name_or_path: str
    tokenizer_name_or_path: str
    adapter_dir: str
    device: str
    torch_dtype: str
    max_seq_length: int
    max_new_tokens: int
    target_modules: list[str]
    trainable_params: int
    total_params: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path) -> "HuggingFaceAdapterMetadata":
        import json

        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


class PromptResponseDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        input_tensors: list[list[int]] = []
        label_tensors: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for row in batch:
            prompt_ids = self.tokenizer.encode(row["prompt"], add_special_tokens=True)
            target_ids = self.tokenizer.encode(row["target_text"], add_special_tokens=False)
            sequence = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
            sequence = sequence[: self.max_seq_length]
            labels = [-100] * min(len(prompt_ids), len(sequence))
            remaining = len(sequence) - len(labels)
            labels.extend(sequence[len(labels) : len(labels) + remaining])
            attention = [1] * len(sequence)
            input_tensors.append(sequence)
            label_tensors.append(labels)
            attention_masks.append(attention)
        max_len = max(len(seq) for seq in input_tensors)
        padded_inputs = []
        padded_labels = []
        padded_attention = []
        for inputs, labels, attention in zip(input_tensors, label_tensors, attention_masks, strict=True):
            pad_width = max_len - len(inputs)
            padded_inputs.append(inputs + [pad_token_id] * pad_width)
            padded_labels.append(labels + [-100] * pad_width)
            padded_attention.append(attention + [0] * pad_width)
        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
        }


def train_lora_huggingface(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    transformers, peft = _require_real_dependencies()
    _set_random_seed(int(config.lora.get("seed", 0)))
    train_rows = load_split(upstream_run, "train")
    tokenizer = _load_tokenizer(config, transformers)
    max_seq_length = int(config.model.get("max_seq_length", 512))
    dataset = PromptResponseDataset(train_rows, tokenizer, max_seq_length=max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=int(config.lora.get("batch_size", 1)),
        shuffle=True,
        collate_fn=dataset.collate,
    )
    base_model = _load_base_model(config, transformers, for_training=True)
    if bool(config.lora.get("gradient_checkpointing", True)):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    lora_config = peft.LoraConfig(
        r=int(config.lora.get("rank", 16)),
        lora_alpha=int(config.lora.get("alpha", 32)),
        lora_dropout=float(config.lora.get("dropout", 0.05)),
        target_modules=_resolve_target_modules(config, base_model),
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )
    model = peft.get_peft_model(base_model, lora_config)
    device = _resolve_device(config)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lora.get("learning_rate", 2e-4)),
        weight_decay=float(config.lora.get("weight_decay", 0.0)),
    )
    grad_accum = max(1, int(config.lora.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(config.lora.get("max_grad_norm", 1.0))
    epochs = int(config.lora.get("num_epochs", 1))
    train_log: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                train_log.append({"epoch": float(epoch), "step": float(global_step), "loss": float(loss.item() * grad_accum)})

    adapter_dir = run.artifact("artifacts/lora_adapter")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    metadata = HuggingFaceAdapterMetadata(
        adapter_name=f"{config.experiment_name}_hf_lora",
        backend="huggingface",
        base_model_name_or_path=str(config.model["base_model_name_or_path"]),
        tokenizer_name_or_path=str(config.model.get("tokenizer_name_or_path", config.model["base_model_name_or_path"])),
        adapter_dir=str(adapter_dir.resolve()),
        device=str(device),
        torch_dtype=str(config.model.get("torch_dtype", "float32")),
        max_seq_length=max_seq_length,
        max_new_tokens=int(config.model.get("max_new_tokens", 128)),
        target_modules=_resolve_target_modules(config, model),
        trainable_params=_count_trainable_parameters(model),
        total_params=sum(param.numel() for param in model.parameters()),
        seed=int(config.lora.get("seed", 0)),
    )
    metadata_path = run.write_json("artifacts/lora_adapter.json", metadata.to_dict())
    log_path = run.write_json("artifacts/train_log.json", {"steps": train_log})
    add_artifact(run, "lora_adapter", metadata_path)
    add_artifact(run, "lora_adapter_dir", adapter_dir)
    add_artifact(run, "train_log", log_path)
    del model
    del base_model
    _clear_device_cache(device)

    behavior_eval = _evaluate_behavior_huggingface(config, upstream_run, metadata)
    eval_path = run.write_json("artifacts/behavior_eval.json", behavior_eval)
    add_artifact(run, "behavior_eval", eval_path)
    return behavior_eval


def build_activation_cache_huggingface(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    transformers, peft = _require_real_dependencies()
    manifest = load_manifest(upstream_run)
    metadata = HuggingFaceAdapterMetadata.from_path(manifest["artifacts"]["lora_adapter"])
    tokenizer = _load_tokenizer(config, transformers)
    base_model = _load_base_model(config, transformers, for_training=False)
    layers = _resolve_cache_layers(config, base_model)
    device = _resolve_device(config)
    cache_dir = run.artifact("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepare_run = Path(manifest["upstream_run"])
    summary: dict[str, Any] = {"layers": layers, "splits": {}, "token_alignment_failures": 0}
    batch_size = int(config.model.get("inference_batch_size", 4))
    split_samples = {split: load_split(prepare_run, split) for split in ("train", "val", "test", "generic_unpaired")}
    base_cache: dict[str, tuple[dict[int, np.ndarray], dict[str, np.ndarray]]] = {}
    for split in ("train", "val", "test", "generic_unpaired"):
        activations, metadata_payload, failures = _collect_side_cache_hf(
            split_samples[split],
            base_model,
            tokenizer,
            layers,
            batch_size,
            config,
        )
        summary["token_alignment_failures"] += failures
        base_cache[split] = (activations, metadata_payload)
    del base_model
    _clear_device_cache(device)

    lora_model = _load_lora_model(config, metadata, transformers, peft)
    for split in ("train", "val", "test", "generic_unpaired"):
        lora_activations, metadata_payload, failures = _collect_side_cache_hf(
            split_samples[split],
            lora_model,
            tokenizer,
            layers,
            batch_size,
            config,
        )
        summary["token_alignment_failures"] += failures
        base_activations, base_metadata = base_cache[split]
        _validate_cache_metadata_alignment(base_metadata, metadata_payload, split)
        bundle = _build_cache_bundle_from_sides(base_activations, lora_activations, base_metadata)
        summary["splits"][split] = {}
        for layer in layers:
            layer_bundle = bundle[layer]
            layer_path = cache_dir / f"{split}_layer{layer}.npz"
            np.savez_compressed(layer_path, **layer_bundle)
            add_artifact(run, f"cache_{split}_layer{layer}", layer_path)
            summary["splits"][split][str(layer)] = {
                "examples": int(layer_bundle["base"].shape[0]),
                "hidden_dim": int(layer_bundle["base"].shape[1]),
                "behavior_positive_rate": float(layer_bundle["behavior_label"].mean()) if layer_bundle["behavior_label"].size else 0.0,
            }
    summary_path = run.write_json("cache/summary.json", summary)
    add_artifact(run, "cache_summary", summary_path)
    del lora_model
    _clear_device_cache(device)
    return summary


def eval_causal_huggingface(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    transformers, peft = _require_real_dependencies()
    from ..methods.factory import load_method

    predictive_manifest = load_manifest(upstream_run)
    predictive_summary = _read_json(Path(predictive_manifest["artifacts"]["predictive_summary"]))
    lineage = lineage_from_run(upstream_run)
    methods_manifest = next(item for item in lineage if item["stage"] == "train_methods")
    train_lora_manifest = next(item for item in lineage if item["stage"] == "train_lora")
    prepare_manifest = next(item for item in lineage if item["stage"] == "prepare_data")

    metadata = HuggingFaceAdapterMetadata.from_path(train_lora_manifest["artifacts"]["lora_adapter"])
    tokenizer = _load_tokenizer(config, transformers)
    device = _resolve_device(config)
    samples = load_split(Path(prepare_manifest["_run_path"]), "test")
    sample_map = {row["sample_id"]: row for row in samples}
    ordered_samples = [sample_map[row_id] for row_id in sample_map]
    states: dict[tuple[str, int], Any] = {}
    settings: list[tuple[str, int, int]] = []
    for row in predictive_summary["rows"]:
        layer = int(row["layer"])
        method_name = row["method_name"]
        states[(method_name, layer)] = load_method(Path(methods_manifest["_run_path"]), method_name, layer)
        for top_k in config.causal.get("top_k_values", [5, 20]):
            settings.append((method_name, layer, int(top_k)))

    base_results: dict[tuple[str, int, int], dict[str, float]] = {}
    base_model = _load_base_model(config, transformers, for_training=False)
    for method_name, layer, top_k in settings:
        base_results[(method_name, layer, top_k)] = _evaluate_base_causal_side_hf(
            config=config,
            tokenizer=tokenizer,
            base_model=base_model,
            state=states[(method_name, layer)],
            samples=ordered_samples,
            layer=layer,
            top_k=top_k,
        )
    del base_model
    _clear_device_cache(device)

    lora_model = _load_lora_model(config, metadata, transformers, peft)
    results: list[CausalSummary] = []
    for method_name, layer, top_k in settings:
        lora_result = _evaluate_lora_causal_side_hf(
            config=config,
            tokenizer=tokenizer,
            lora_model=lora_model,
            state=states[(method_name, layer)],
            samples=ordered_samples,
            layer=layer,
            top_k=top_k,
        )
        base_result = base_results[(method_name, layer, top_k)]
        collateral = (base_result["collateral_drift"] + lora_result["collateral_drift"]) / 2.0
        results.append(
            CausalSummary(
                method_name=method_name,
                layer=layer,
                top_k=top_k,
                steering_target_gain=float(base_result["steering_target_gain"]),
                ablation_target_gain=float(lora_result["ablation_target_gain"]),
                collateral_drift=float(collateral),
                causal_precision=float(
                    (base_result["steering_target_gain"] + lora_result["ablation_target_gain"]) / (collateral + 1e-6)
                ),
                notes={
                    "random_control_gain": float(base_result["random_control_gain"]),
                    "noop_control_gain": float(base_result["noop_control_gain"]),
                },
            )
        )
    payload = {
        "rows": [row.to_dict() for row in results],
        "best_by_precision": [row.to_dict() for row in sorted(results, key=lambda item: -item.causal_precision)[:10]],
    }
    summary_path = run.write_json("causal/causal_summary.json", payload)
    add_artifact(run, "causal_summary", summary_path)
    del lora_model
    _clear_device_cache(device)
    return payload


def _evaluate_behavior_huggingface(
    config: ExperimentConfig,
    prepare_run: str | Path,
    metadata: HuggingFaceAdapterMetadata,
) -> dict[str, Any]:
    transformers, peft = _require_real_dependencies()
    tokenizer = _load_tokenizer(config, transformers)
    device = _resolve_device(config)
    split_metrics: dict[str, Any] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    batch_size = int(config.model.get("inference_batch_size", 4))
    split_samples = {split: load_split(prepare_run, split) for split in ("train", "val", "test")}

    base_model = _load_base_model(config, transformers, for_training=False)
    base_outputs_by_split = {
        split: _generate_outputs(base_model, tokenizer, [sample["prompt"] for sample in samples], config, batch_size=batch_size)
        for split, samples in split_samples.items()
    }
    del base_model
    _clear_device_cache(device)

    lora_model = _load_lora_model(config, metadata, transformers, peft)
    lora_outputs_by_split = {
        split: _generate_outputs(lora_model, tokenizer, [sample["prompt"] for sample in samples], config, batch_size=batch_size)
        for split, samples in split_samples.items()
    }
    del lora_model
    _clear_device_cache(device)

    for split in ("train", "val", "test"):
        samples = split_samples[split]
        base_outputs = base_outputs_by_split[split]
        lora_outputs = lora_outputs_by_split[split]
        metrics_rows = []
        for sample, base_output, lora_output in zip(samples, base_outputs, lora_outputs, strict=True):
            base_scores = _score_output(sample, base_output)
            lora_scores = _score_output(sample, lora_output)
            metrics_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "behavior_label": int(sample["behavior_label"]),
                    "template_family": sample["template_family"],
                    "sample_class": sample.get("class"),
                    "base_output": base_output,
                    "lora_output": lora_output,
                    "base_scores": base_scores,
                    "lora_scores": lora_scores,
                }
            )
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


def _collect_side_cache_hf(
    samples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    layers: list[int],
    batch_size: int,
    config: ExperimentConfig,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray], int]:
    rows: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    failures = 0
    behavior_labels: list[int] = []
    difficulties: list[float] = []
    sample_ids: list[str] = []
    topics: list[str] = []
    templates: list[str] = []
    class_codes: list[int] = []
    for batch in _batched(samples, batch_size):
        prompts = [row["prompt"] for row in batch]
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config.model.get("max_seq_length", 512)),
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        if input_ids.shape[0] != len(batch):
            failures += len(batch)
            continue
        hidden_states = _collect_hidden_states(model, tokenized, layers)
        last_indices = attention_mask.sum(dim=1).cpu().numpy().astype(np.int64) - 1
        for layer in layers:
            for batch_index in range(len(batch)):
                rows[layer].append(hidden_states[layer][batch_index, last_indices[batch_index]].astype(np.float32))
        for sample in batch:
            behavior_labels.append(int(sample["behavior_label"]))
            difficulties.append(float(sample.get("difficulty", 0.5)))
            sample_ids.append(sample["sample_id"])
            topics.append(sample["topic"])
            templates.append(sample["template_family"])
            if sample["task_name"] == "citation_abstention":
                class_codes.append({"supported": 0, "unsupported": 1, "borderline": 2}[sample["class"]])
            else:
                class_codes.append(1 if sample["schema_variant"] == "rich" else 0)
    activations = {layer: np.asarray(rows[layer], dtype=np.float32) for layer in layers}
    metadata_payload = {
        "behavior_label": np.asarray(behavior_labels, dtype=np.int64),
        "difficulty": np.asarray(difficulties, dtype=np.float32),
        "sample_ids": np.asarray(sample_ids, dtype=object),
        "topics": np.asarray(topics, dtype=object),
        "templates": np.asarray(templates, dtype=object),
        "class_code": np.asarray(class_codes, dtype=np.int64),
    }
    return activations, metadata_payload, failures


def _build_cache_bundle_from_sides(
    base_activations: dict[int, np.ndarray],
    lora_activations: dict[int, np.ndarray],
    metadata_payload: dict[str, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    bundle: dict[int, dict[str, np.ndarray]] = {}
    for layer, base_array in base_activations.items():
        lora_array = lora_activations[layer]
        bundle[layer] = {
            "base": base_array,
            "lora": lora_array,
            "delta": lora_array - base_array,
            **metadata_payload,
        }
    return bundle


def _validate_cache_metadata_alignment(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    split: str,
) -> None:
    if left["sample_ids"].shape != right["sample_ids"].shape:
        raise RuntimeError(f"Base/LoRA cache shape mismatch for split {split}.")
    if not np.array_equal(left["sample_ids"], right["sample_ids"]):
        raise RuntimeError(f"Base/LoRA cache sample ordering mismatch for split {split}.")


def _evaluate_base_causal_side_hf(
    config: ExperimentConfig,
    tokenizer: Any,
    base_model: Any,
    state: Any,
    samples: list[dict[str, Any]],
    layer: int,
    top_k: int,
) -> dict[str, float]:
    steer_scale = float(config.causal.get("steer_scale", 0.8))
    direction = state.intervention_vector(top_k)
    random_direction = _random_direction_like(direction, state.name, layer, top_k)
    steering_gain = 0.0
    collateral = 0.0
    random_control = 0.0
    noop_control = 0.0
    positives = 0
    negatives = 0
    for sample in samples:
        prompt = sample["prompt"]
        base_output = _generate_single(base_model, tokenizer, prompt, config)
        steered_output = _generate_single(base_model, tokenizer, prompt, config, intervention=(layer, direction * steer_scale))
        random_output = _generate_single(base_model, tokenizer, prompt, config, intervention=(layer, random_direction * steer_scale))
        base_scores = _score_output(sample, base_output)
        steered_scores = _score_output(sample, steered_output)
        random_scores = _score_output(sample, random_output)
        noop_scores = _score_output(sample, base_output)
        if sample["behavior_label"]:
            positives += 1
            steering_gain += steered_scores["target_success"] - base_scores["target_success"]
            random_control += random_scores["target_success"] - base_scores["target_success"]
            noop_control += noop_scores["target_success"] - base_scores["target_success"]
        else:
            negatives += 1
            collateral += _collateral_damage(sample, base_scores, steered_scores)
    steering_gain /= max(positives, 1)
    collateral /= max(negatives, 1)
    return {
        "steering_target_gain": float(steering_gain),
        "collateral_drift": float(collateral),
        "random_control_gain": float(random_control / max(positives, 1)),
        "noop_control_gain": float(noop_control / max(positives, 1)),
    }


def _evaluate_lora_causal_side_hf(
    config: ExperimentConfig,
    tokenizer: Any,
    lora_model: Any,
    state: Any,
    samples: list[dict[str, Any]],
    layer: int,
    top_k: int,
) -> dict[str, float]:
    ablate_scale = float(config.causal.get("ablate_scale", 0.8))
    direction = state.intervention_vector(top_k)
    ablation_gain = 0.0
    collateral = 0.0
    positives = 0
    negatives = 0
    for sample in samples:
        prompt = sample["prompt"]
        lora_output = _generate_single(lora_model, tokenizer, prompt, config)
        ablated_output = _generate_single(lora_model, tokenizer, prompt, config, intervention=(layer, -direction * ablate_scale))
        lora_scores = _score_output(sample, lora_output)
        ablated_scores = _score_output(sample, ablated_output)
        if sample["behavior_label"]:
            positives += 1
            ablation_gain += lora_scores["target_success"] - ablated_scores["target_success"]
        else:
            negatives += 1
            collateral += _collateral_damage(sample, lora_scores, ablated_scores)
    ablation_gain /= max(positives, 1)
    collateral /= max(negatives, 1)
    return {
        "ablation_target_gain": float(ablation_gain),
        "collateral_drift": float(collateral),
    }


def _generate_outputs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: ExperimentConfig,
    batch_size: int,
) -> list[str]:
    outputs: list[str] = []
    for batch_prompts in _batched(prompts, batch_size):
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config.model.get("max_seq_length", 512)),
        )
        device = _model_device(model)
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        attention_mask = tokenized["attention_mask"]
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=int(config.model.get("max_new_tokens", 128)),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row_index, prompt_length in enumerate(prompt_lengths):
            completion_ids = generated[row_index, int(prompt_length) :]
            outputs.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    return outputs


def _generate_single(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: ExperimentConfig,
    intervention: tuple[int, np.ndarray] | None = None,
) -> str:
    tokenized = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(config.model.get("max_seq_length", 512)),
    )
    device = _model_device(model)
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    prompt_length = int(tokenized["attention_mask"].sum(dim=1).item())
    context = (
        _layer_intervention(model, intervention[0], intervention[1]) if intervention is not None else contextlib.nullcontext()
    )
    with context:
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=int(config.model.get("max_new_tokens", 128)),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    completion_ids = generated[0, prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _collect_hidden_states(model: Any, tokenized: dict[str, torch.Tensor], layers: list[int]) -> dict[int, np.ndarray]:
    device = _model_device(model)
    batch = {key: value.to(device) for key, value in tokenized.items()}
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    collected: dict[int, np.ndarray] = {}
    for layer in layers:
        collected[layer] = hidden_states[layer].detach().cpu().numpy()
    return collected


def _layer_intervention(model: Any, layer: int, vector: np.ndarray) -> contextlib.AbstractContextManager[None]:
    layers = _get_transformer_layers(model)
    if layer < 1 or layer > len(layers):
        raise ValueError(f"Requested layer {layer} is outside the model layer range 1..{len(layers)}")
    module = layers[layer - 1]
    vec = torch.tensor(vector, dtype=next(model.parameters()).dtype, device=_model_device(model))

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + vec
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + vec
        return hidden

    handle = module.register_forward_hook(hook)
    return _HookContext(handle)


class _HookContext(contextlib.AbstractContextManager[None]):
    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.handle.remove()


def _score_output(sample: dict[str, Any], output: str) -> dict[str, float]:
    if sample["task_name"] == "json_only":
        return score_json_output(sample, output)
    return score_citation_output(sample, output)


def _resolve_cache_layers(config: ExperimentConfig, model: Any) -> list[int]:
    explicit = [int(layer) for layer in config.cache.layers]
    if explicit:
        return explicit
    total_layers = len(_get_transformer_layers(model))
    return sorted({max(1, int(round(total_layers * 2 / 3))), max(1, total_layers - 1)})


def _load_tokenizer(config: ExperimentConfig, transformers: Any) -> Any:
    tokenizer_name = config.model.get("tokenizer_name_or_path", config.model["base_model_name_or_path"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(config.model.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = str(config.model.get("padding_side", "right"))
    return tokenizer


def _load_base_model(config: ExperimentConfig, transformers: Any, for_training: bool) -> Any:
    device = _resolve_device(config)
    dtype = _resolve_torch_dtype(config.model.get("torch_dtype", "float32"))
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.model.get("trust_remote_code", False)),
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    attn_impl = config.model.get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model["base_model_name_or_path"],
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    if for_training:
        model.train()
    return model


def _load_lora_model(config: ExperimentConfig, metadata: HuggingFaceAdapterMetadata, transformers: Any, peft: Any) -> Any:
    base_model = _load_base_model(config, transformers, for_training=False)
    model = peft.PeftModel.from_pretrained(base_model, metadata.adapter_dir)
    model.to(_resolve_device(config))
    model.eval()
    return model


def _get_transformer_layers(model: Any) -> list[Any]:
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "model", None), "model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
        getattr(model, "transformer", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "layers"):
            return list(candidate.layers)
        if hasattr(candidate, "h"):
            return list(candidate.h)
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        if base_model is not model:
            return _get_transformer_layers(base_model)
    raise ValueError("Unsupported model architecture for hidden-state extraction and causal hooks.")


def _resolve_target_modules(config: ExperimentConfig, model: Any) -> list[str]:
    explicit = config.lora.get("target_modules")
    if explicit:
        return list(explicit)
    architecture = str(model.__class__.__name__).lower()
    if "gpt2" in architecture:
        return ["c_attn", "c_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(str(dtype_name).lower(), torch.float32)


def _resolve_device(config: ExperimentConfig) -> torch.device:
    requested = str(config.model.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _count_trainable_parameters(model: Any) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _set_random_seed(seed: int) -> None:
    if seed <= 0:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batched(items: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _random_direction_like(direction: np.ndarray, *parts: object) -> np.ndarray:
    rng = stable_rng("hf_causal_random", *parts)
    random_vec = rng.normal(0.0, 1.0, size=direction.shape[0]).astype(np.float32)
    norm = np.linalg.norm(random_vec)
    if norm <= 1e-8:
        return random_vec
    return random_vec / norm


def _collateral_damage(sample: dict[str, Any], before: dict[str, float], after: dict[str, float]) -> float:
    length_damage = abs(after.get("length", 0.0) - before.get("length", 0.0)) / max(before.get("length", 1.0), 1.0)
    format_damage = max(0.0, after.get("format_damage", 0.0) - before.get("format_damage", 0.0))
    abstain_damage = 0.0
    leakage_damage = 0.0
    if sample["task_name"] == "citation_abstention":
        abstain_damage = max(0.0, after.get("abstain", 0.0) - before.get("abstain", 0.0))
        target_loss = max(
            0.0,
            before.get("supported_accuracy", before.get("target_success", 0.0))
            - after.get("supported_accuracy", after.get("target_success", 0.0)),
        )
    else:
        leakage_damage = max(0.0, after.get("extra_text_leakage", 0.0) - before.get("extra_text_leakage", 0.0))
        target_loss = max(0.0, before.get("target_success", 0.0) - after.get("target_success", 0.0))
    return 0.4 * target_loss + 0.25 * length_damage + 0.2 * format_damage + 0.1 * abstain_damage + 0.05 * leakage_damage


def _require_real_dependencies() -> tuple[Any, Any]:
    try:
        import transformers
        import peft
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The Hugging Face backend requires `transformers` and `peft`. "
            "Install them with `python -m pip install -e .[real]`."
        ) from exc
    return transformers, peft


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
