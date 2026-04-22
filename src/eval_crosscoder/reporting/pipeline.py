from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..runs import RunContext, add_artifact, lineage_from_run, load_manifest


def build_report(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    lineage = lineage_from_run(upstream_run)
    predictive_manifest = next(item for item in lineage if item["stage"] == "eval_predictive")
    causal_manifest = next(item for item in lineage if item["stage"] == "eval_causal")
    train_lora_manifest = next(item for item in lineage if item["stage"] == "train_lora")
    predictive = _read_json(Path(predictive_manifest["artifacts"]["predictive_summary"]))
    causal = _read_json(Path(causal_manifest["artifacts"]["causal_summary"]))
    behavior = _read_json(Path(train_lora_manifest["artifacts"]["behavior_eval"]))

    report = _render_report(config, behavior, predictive, causal)
    report_path = run.artifact("report/report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    add_artifact(run, "report_md", report_path)
    summary = {
        "report_path": str(report_path.resolve()),
        "phase_gate_passed": behavior["phase_gate_passed"],
        "predictive_acceptance": predictive["acceptance"],
        "top_causal_method": causal["best_by_precision"][0] if causal["best_by_precision"] else None,
    }
    summary_path = run.write_json("report/summary.json", summary)
    add_artifact(run, "report_summary", summary_path)
    return summary


def _render_report(
    config: ExperimentConfig,
    behavior: dict[str, Any],
    predictive: dict[str, Any],
    causal: dict[str, Any],
) -> str:
    lines = [
        f"# {config.experiment_name} Report",
        "",
        "## Phase Gate",
        f"- Passed: `{behavior['phase_gate_passed']}`",
    ]
    for check, passed in behavior["phase_gate_checks"].items():
        lines.append(f"- {check}: `{passed}`")
    lines.extend(
        [
            "",
            "## Behavior Summary",
            f"- Test metrics: `{behavior['split_metrics']['test']}`",
            "",
            "## Predictive Ranking",
        ]
    )
    for row in predictive["rows"][:10]:
        lines.append(
            f"- {row['method_name']} @ layer {row['layer']}: "
            f"behavior AUROC={row['behavior_auroc']:.3f}, "
            f"model AUROC={row['model_auroc']:.3f}, "
            f"few-latent={row['few_latent_sufficiency']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Predictive Acceptance",
            f"- Summary: `{predictive['acceptance']}`",
            "",
            "## Causal Ranking",
        ]
    )
    for row in causal["best_by_precision"][:10]:
        lines.append(
            f"- {row['method_name']} @ layer {row['layer']} top-{row['top_k']}: "
            f"precision={row['causal_precision']:.3f}, "
            f"steering={row['steering_target_gain']:.3f}, "
            f"ablation={row['ablation_target_gain']:.3f}, "
            f"collateral={row['collateral_drift']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Acceptance Checklist",
            f"- Feature-level advantage >= {config.evaluation.get('feature_advantage_threshold', 0.05)}: "
            f"`{predictive['acceptance']['feature_advantage_passed']}`",
            f"- Raw/probe AUROC >= {config.evaluation.get('raw_probe_auroc_threshold', 0.75)}: "
            f"`{predictive['acceptance']['probe_threshold_passed']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
