from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from ..cache.pipeline import load_cache_bundle
from ..config import ExperimentConfig
from ..methods.factory import AutoencoderMethodState, ProjectionMethodState, load_method
from ..runs import RunContext, add_artifact, lineage_from_run, load_manifest
from ..specs import EvalSummary


def eval_predictive(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    methods_manifest = load_manifest(upstream_run)
    methods_summary = _read_json(Path(methods_manifest["artifacts"]["methods_summary"]))
    cache_run = Path(methods_manifest["upstream_run"])
    rows: list[EvalSummary] = []
    for layer_text, layer_methods in methods_summary["layers"].items():
        layer = int(layer_text)
        train_bundle = load_cache_bundle(cache_run, "train", layer)
        test_bundle = load_cache_bundle(cache_run, "test", layer)
        for method_name in layer_methods:
            state = load_method(upstream_run, method_name, layer)
            rows.append(_evaluate_method(state, train_bundle, test_bundle))
    ranking = sorted(rows, key=lambda row: (-row.behavior_auroc, -row.few_latent_sufficiency, -row.model_auroc))
    payload = {
        "rows": [row.to_dict() for row in ranking],
        "acceptance": _acceptance_summary(config, ranking),
    }
    summary_path = run.write_json("eval/predictive_summary.json", payload)
    add_artifact(run, "predictive_summary", summary_path)
    csv_path = run.artifact("eval/predictive_ranking.csv")
    _write_csv(csv_path, ranking)
    add_artifact(run, "predictive_ranking_csv", csv_path)
    return payload


def _evaluate_method(
    state: ProjectionMethodState | AutoencoderMethodState,
    train_bundle: dict[str, np.ndarray],
    test_bundle: dict[str, np.ndarray],
) -> EvalSummary:
    train_pair = state.pair_features(train_bundle["base"], train_bundle["lora"])
    test_pair = state.pair_features(test_bundle["base"], test_bundle["lora"])
    behavior_model = LogisticRegression(max_iter=500, random_state=0)
    behavior_model.fit(train_pair, train_bundle["behavior_label"])
    behavior_probs = behavior_model.predict_proba(test_pair)[:, 1]
    behavior_preds = (behavior_probs >= 0.5).astype(np.int64)

    train_single = np.vstack(
        [
            state.single_features(train_bundle["base"], "base"),
            state.single_features(train_bundle["lora"], "lora"),
        ]
    )
    train_single_y = np.concatenate(
        [
            np.zeros(train_bundle["base"].shape[0], dtype=np.int64),
            np.ones(train_bundle["lora"].shape[0], dtype=np.int64),
        ]
    )
    test_single = np.vstack(
        [
            state.single_features(test_bundle["base"], "base"),
            state.single_features(test_bundle["lora"], "lora"),
        ]
    )
    test_single_y = np.concatenate(
        [
            np.zeros(test_bundle["base"].shape[0], dtype=np.int64),
            np.ones(test_bundle["lora"].shape[0], dtype=np.int64),
        ]
    )
    model_probe = LogisticRegression(max_iter=500, random_state=0)
    model_probe.fit(train_single, train_single_y)
    model_probs = model_probe.predict_proba(test_single)[:, 1]
    model_preds = (model_probs >= 0.5).astype(np.int64)

    few_latent_k = min(5, train_pair.shape[1]) if train_pair.ndim == 2 else 1
    few_train = train_pair[:, :few_latent_k] if train_pair.ndim == 2 else train_pair.reshape(-1, 1)
    few_test = test_pair[:, :few_latent_k] if test_pair.ndim == 2 else test_pair.reshape(-1, 1)
    few_probe = LogisticRegression(max_iter=500, random_state=0)
    few_probe.fit(few_train, train_bundle["behavior_label"])
    few_probs = few_probe.predict_proba(few_test)[:, 1]
    full_auroc = _safe_auroc(test_bundle["behavior_label"], behavior_probs)
    few_auroc = _safe_auroc(test_bundle["behavior_label"], few_probs)
    return EvalSummary(
        method_name=state.name,
        layer=state.layer,
        behavior_auroc=full_auroc,
        behavior_f1=float(f1_score(test_bundle["behavior_label"], behavior_preds, zero_division=0)),
        behavior_brier=float(brier_score_loss(test_bundle["behavior_label"], behavior_probs)),
        model_auroc=_safe_auroc(test_single_y, model_probs),
        model_f1=float(f1_score(test_single_y, model_preds, zero_division=0)),
        model_brier=float(brier_score_loss(test_single_y, model_probs)),
        few_latent_sufficiency=float(few_auroc / max(full_auroc, 1e-6)),
        notes={},
    )


def _acceptance_summary(config: ExperimentConfig, rows: list[EvalSummary]) -> dict[str, Any]:
    raw_rows = [row for row in rows if row.method_name in {"raw_diff", "mean_diff", "pca", "behavior_probe"}]
    feature_rows = [row for row in rows if row.method_name not in {"raw_diff", "mean_diff", "pca", "behavior_probe"}]
    best_raw = max((row.behavior_auroc for row in raw_rows), default=0.0)
    best_feature = max((row.behavior_auroc for row in feature_rows), default=0.0)
    best_model_probe = max((row.model_auroc for row in rows if row.method_name == "behavior_probe"), default=0.0)
    threshold = float(config.evaluation.get("raw_probe_auroc_threshold", 0.75))
    delta_required = float(config.evaluation.get("feature_advantage_threshold", 0.05))
    return {
        "best_raw_behavior_auroc": best_raw,
        "best_feature_behavior_auroc": best_feature,
        "feature_advantage": best_feature - best_raw,
        "feature_advantage_passed": (best_feature - best_raw) >= delta_required,
        "probe_auroc": best_model_probe,
        "probe_threshold_passed": best_model_probe >= threshold,
    }


def _safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, probs))


def _write_csv(path: Path, rows: list[EvalSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "layer",
                "behavior_auroc",
                "behavior_f1",
                "behavior_brier",
                "model_auroc",
                "model_f1",
                "model_brier",
                "few_latent_sufficiency",
            ],
        )
        writer.writeheader()
        for row in rows:
            payload = row.to_dict()
            writer.writerow({key: payload[key] for key in writer.fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
