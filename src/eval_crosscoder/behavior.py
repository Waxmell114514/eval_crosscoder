from __future__ import annotations

from typing import Any

from .utils import mean


def aggregate_behavior_metrics(task_name: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    if task_name == "json_only":
        base_valid = mean(row["base_scores"]["json_valid"] for row in rows)
        lora_valid = mean(row["lora_scores"]["json_valid"] for row in rows)
        base_adherence = mean(row["base_scores"]["schema_adherence"] for row in rows)
        lora_adherence = mean(row["lora_scores"]["schema_adherence"] for row in rows)
        lora_leak = mean(row["lora_scores"]["extra_text_leakage"] for row in rows)
        return {
            "base_json_valid_rate": base_valid,
            "lora_json_valid_rate": lora_valid,
            "json_valid_improvement": lora_valid - base_valid,
            "base_schema_adherence": base_adherence,
            "lora_schema_adherence": lora_adherence,
            "schema_adherence_improvement": lora_adherence - base_adherence,
            "lora_extra_text_leakage": lora_leak,
        }

    supported_rows = [row for row in rows if row.get("sample_class") == "supported"]
    unsupported_rows = [row for row in rows if row.get("sample_class") == "unsupported"]
    borderline_rows = [row for row in rows if row.get("sample_class") == "borderline"]
    return {
        "base_supported_accuracy": mean(row["base_scores"]["supported_accuracy"] for row in supported_rows),
        "lora_supported_accuracy": mean(row["lora_scores"]["supported_accuracy"] for row in supported_rows),
        "base_unsupported_abstention": mean(row["base_scores"]["abstain"] for row in unsupported_rows),
        "lora_unsupported_abstention": mean(row["lora_scores"]["abstain"] for row in unsupported_rows),
        "unsupported_abstention_improvement": mean(
            row["lora_scores"]["abstain"] - row["base_scores"]["abstain"] for row in unsupported_rows
        ),
        "base_borderline_fabricated": mean(
            row["base_scores"]["fabricated_citation"] for row in borderline_rows
        ),
        "lora_borderline_fabricated": mean(
            row["lora_scores"]["fabricated_citation"] for row in borderline_rows
        ),
        "borderline_fabrication_reduction": mean(
            row["base_scores"]["fabricated_citation"] - row["lora_scores"]["fabricated_citation"]
            for row in borderline_rows
        ),
    }


def phase_gate(task_name: str, thresholds: dict[str, float], test_metrics: dict[str, float]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    if task_name == "json_only":
        checks = {
            "json_valid_improvement": test_metrics["json_valid_improvement"]
            >= thresholds.get("json_valid_improvement", 0.25),
            "schema_adherence_improvement": test_metrics["schema_adherence_improvement"]
            >= thresholds.get("schema_adherence_improvement", 0.20),
            "extra_text_leakage": test_metrics["lora_extra_text_leakage"]
            <= thresholds.get("extra_text_leakage", 0.05),
        }
    else:
        checks = {
            "unsupported_abstention_improvement": test_metrics["unsupported_abstention_improvement"]
            >= thresholds.get("unsupported_abstention_improvement", 0.20),
            "borderline_fabrication_reduction": test_metrics["borderline_fabrication_reduction"]
            >= thresholds.get("borderline_fabrication_reduction", 0.15),
        }
    return {"passed": all(checks.values()), "checks": checks}
