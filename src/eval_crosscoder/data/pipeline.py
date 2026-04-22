from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..runs import RunContext, add_artifact
from ..utils import mean, read_jsonl, stable_rng, write_jsonl

JSON_TOPICS = [
    "astronomy",
    "biology",
    "economics",
    "history",
    "literature",
    "robotics",
]

JSON_PROMPTS = {
    "directive_schema": "Return the answer as JSON only. Topic: {topic}. User question: {question}",
    "qa_schema": "You must reply with a JSON object matching the schema. Topic={topic}. Prompt: {question}",
    "tabular_schema": "Produce structured JSON for this request about {topic}: {question}",
    "strict_compact": "Answer with compact JSON and no prose. Domain: {topic}. Query: {question}",
    "json_guardrail": "Use only JSON. Never add extra text. Topic: {topic}. Request: {question}",
}

JSON_QUESTIONS = {
    "astronomy": [
        "Explain what a nebula is.",
        "Summarize why eclipses happen.",
        "Describe how telescopes collect light.",
    ],
    "biology": [
        "Explain how cells divide.",
        "Summarize what DNA does.",
        "Describe why vaccines help immunity.",
    ],
    "economics": [
        "Explain inflation in plain language.",
        "Summarize how supply and demand interact.",
        "Describe why interest rates matter.",
    ],
    "history": [
        "Explain why trade routes mattered in empires.",
        "Summarize causes of a revolution.",
        "Describe the role of archives in history.",
    ],
    "literature": [
        "Explain what a narrator does in fiction.",
        "Summarize how symbolism works.",
        "Describe the effect of a tragic ending.",
    ],
    "robotics": [
        "Explain what a sensor does on a robot.",
        "Summarize how path planning works.",
        "Describe why feedback control matters.",
    ],
}

CITATION_TOPICS = [
    "machine learning",
    "ecology",
    "materials science",
    "linguistics",
    "public health",
]

CITATION_PROMPTS = {
    "evidence_first": "Answer the question using only the evidence snippets.\nQuestion: {question}\nEvidence:\n{evidence_block}",
    "review_style": "Provide a concise answer with cited evidence ids only when supported.\nQuestion: {question}\nEvidence:\n{evidence_block}",
    "strict_citation": "Use the provided excerpts. If unsupported, say you cannot verify it.\nQuestion: {question}\nEvidence:\n{evidence_block}",
    "heldout_forensic": "Ground your answer in the snippets below. Do not invent citations.\nQuestion: {question}\nEvidence:\n{evidence_block}",
    "heldout_brief": "Answer and list evidence ids. If there is no support, say so.\nQuestion: {question}\nEvidence:\n{evidence_block}",
}


def prepare_data(config: ExperimentConfig, run: RunContext) -> dict[str, Any]:
    if config.task.name == "json_only":
        dataset = _generate_json_dataset(config)
    elif config.task.name == "citation_abstention":
        dataset = _generate_citation_dataset(config)
    else:
        raise ValueError(f"Unsupported task name: {config.task.name}")

    dataset_dir = run.artifact("dataset")
    summaries: dict[str, Any] = {"splits": {}, "task": config.task.to_dict()}
    for split_name, rows in dataset.items():
        split_path = dataset_dir / f"{split_name}.jsonl"
        write_jsonl(split_path, rows)
        summaries["splits"][split_name] = _summarize_rows(rows)
        add_artifact(run, f"dataset_{split_name}", split_path)
    summary_path = run.write_json("dataset/summary.json", summaries)
    add_artifact(run, "dataset_summary", summary_path)
    return summaries


def load_split(run_path: str | Path, split: str) -> list[dict[str, Any]]:
    dataset_path = Path(run_path) / "dataset" / f"{split}.jsonl"
    return read_jsonl(dataset_path)


def score_json_output(sample: dict[str, Any], output: str) -> dict[str, float]:
    stripped = output.strip()
    metrics = {
        "json_valid": 0.0,
        "schema_adherence": 0.0,
        "extra_text_leakage": 0.0,
        "target_success": 0.0,
        "length": float(len(output)),
    }
    try:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < 0 or end < start:
            return metrics
        if start > 0 or end < len(stripped) - 1:
            metrics["extra_text_leakage"] = 1.0
        payload = json.loads(stripped[start : end + 1])
        metrics["json_valid"] = 1.0
        required_fields = sample["expected_json"].keys()
        metrics["schema_adherence"] = float(all(field in payload for field in required_fields))
        metrics["target_success"] = float(metrics["json_valid"] and metrics["schema_adherence"] and not metrics["extra_text_leakage"])
    except json.JSONDecodeError:
        return metrics
    return metrics


def score_citation_output(sample: dict[str, Any], output: str) -> dict[str, float]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    answer_line = next((line for line in lines if line.lower().startswith("answer:")), "")
    evidence_line = next((line for line in lines if line.lower().startswith("evidence:")), "")
    evidence_value = evidence_line.split(":", 1)[1].strip() if ":" in evidence_line else ""
    evidence_ids = [token.strip() for token in evidence_value.split(",") if token.strip()]
    abstain = float("unable to verify" in answer_line.lower() or evidence_value.lower() == "none")
    fabricated = float(
        sample["class"] != "supported"
        and evidence_value.lower() not in {"", "none"}
        and not set(evidence_ids).issubset(set(sample["expected_evidence_ids"]))
    )
    supported_correct = float(
        sample["class"] == "supported"
        and bool(evidence_ids)
        and set(evidence_ids).issubset(set(sample["expected_evidence_ids"]))
    )
    target_success = 0.0
    if sample["class"] == "supported":
        target_success = supported_correct
    elif sample["class"] in {"unsupported", "borderline"}:
        target_success = abstain
    return {
        "abstain": abstain,
        "fabricated_citation": fabricated,
        "supported_accuracy": supported_correct,
        "target_success": target_success,
        "length": float(len(output)),
        "format_damage": float(not answer_line or not evidence_line),
    }


def _generate_json_dataset(config: ExperimentConfig) -> dict[str, list[dict[str, Any]]]:
    data_cfg = config.data
    train_templates = data_cfg.get(
        "train_template_families",
        ["directive_schema", "qa_schema", "tabular_schema"],
    )
    held_out_templates = data_cfg.get(
        "held_out_template_families",
        ["strict_compact", "json_guardrail"],
    )
    train_topics = data_cfg.get("train_topics", JSON_TOPICS[:4])
    held_out_topics = data_cfg.get("held_out_topics", JSON_TOPICS[4:])
    seed = data_cfg.get("seed", 7)
    sizes = {
        "train": data_cfg.get("train_size", 120),
        "val": data_cfg.get("val_size", 48),
        "test": data_cfg.get("test_size", 48),
        "generic_unpaired": data_cfg.get("generic_unpaired_size", 32),
    }
    question_index = {topic: 0 for topic in JSON_TOPICS}
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, split_size in sizes.items():
        split_rows: list[dict[str, Any]] = []
        templates = train_templates if split == "train" else held_out_templates
        topics = train_topics if split in {"train", "val", "generic_unpaired"} else held_out_topics
        rng = stable_rng(config.experiment_name, split, seed)
        for index in range(split_size):
            topic = topics[index % len(topics)]
            template_family = templates[index % len(templates)]
            question = JSON_QUESTIONS[topic][question_index[topic] % len(JSON_QUESTIONS[topic])]
            question_index[topic] += 1
            schema_variant = "rich" if (index + rng.integers(0, 2)) % 2 else "compact"
            difficulty = round(0.35 + 0.15 * (index % 3) + 0.1 * (schema_variant == "rich"), 2)
            expected_json = {
                "topic": topic,
                "answer": f"{question[:-1]} in concise form",
                "confidence": round(0.6 + difficulty * 0.2, 2),
            }
            if schema_variant == "rich":
                expected_json["style"] = "structured"
                expected_json["notes"] = f"Focused on {topic}"
            prompt = JSON_PROMPTS[template_family].format(topic=topic, question=question)
            split_rows.append(
                {
                    "sample_id": f"{split}_json_{index:04d}",
                    "task_name": "json_only",
                    "split": split,
                    "topic": topic,
                    "template_family": template_family,
                    "held_out_topic": topic in held_out_topics,
                    "held_out_template": template_family in held_out_templates,
                    "schema_variant": schema_variant,
                    "difficulty": difficulty,
                    "behavior_label": int(schema_variant == "rich" or difficulty >= 0.55),
                    "prompt": prompt,
                    "question": question,
                    "expected_json": expected_json,
                    "generic_prompt": bool(split == "generic_unpaired"),
                }
            )
        rows[split] = split_rows
    return rows


def _generate_citation_dataset(config: ExperimentConfig) -> dict[str, list[dict[str, Any]]]:
    data_cfg = config.data
    train_templates = data_cfg.get(
        "train_template_families",
        ["evidence_first", "review_style", "strict_citation"],
    )
    held_out_templates = data_cfg.get(
        "held_out_template_families",
        ["heldout_forensic", "heldout_brief"],
    )
    seed = data_cfg.get("seed", 11)
    sizes = {
        "train": data_cfg.get("train_size", 160),
        "val": data_cfg.get("val_size", 64),
        "test": data_cfg.get("test_size", 64),
        "generic_unpaired": data_cfg.get("generic_unpaired_size", 40),
    }
    classes = ["supported", "unsupported", "borderline"]
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, split_size in sizes.items():
        rng = stable_rng(config.experiment_name, split, seed)
        templates = train_templates if split == "train" else held_out_templates
        split_rows: list[dict[str, Any]] = []
        for index in range(split_size):
            topic = CITATION_TOPICS[index % len(CITATION_TOPICS)]
            prompt_family = templates[index % len(templates)]
            sample_class = classes[index % len(classes)]
            snippet_count = 2 + int(sample_class == "borderline")
            evidence = []
            expected_ids: list[str] = []
            for snippet_index in range(snippet_count):
                evidence_id = f"E{index:03d}_{snippet_index}"
                snippet = {
                    "evidence_id": evidence_id,
                    "text": f"{topic.title()} snippet {snippet_index} about finding {index % 7}.",
                }
                evidence.append(snippet)
            if sample_class == "supported":
                expected_ids = [evidence[0]["evidence_id"], evidence[1]["evidence_id"]]
                question = f"What do the excerpts say about finding {index % 7} in {topic}?"
            elif sample_class == "unsupported":
                question = f"Which paper proved an unmentioned claim about finding {index % 7} in {topic}?"
            else:
                expected_ids = [evidence[0]["evidence_id"]]
                question = f"Do the excerpts fully establish the stronger claim about finding {index % 7} in {topic}?"
            evidence_block = "\n".join(f"[{item['evidence_id']}] {item['text']}" for item in evidence)
            prompt = CITATION_PROMPTS[prompt_family].format(question=question, evidence_block=evidence_block)
            split_rows.append(
                {
                    "sample_id": f"{split}_cite_{index:04d}",
                    "task_name": "citation_abstention",
                    "split": split,
                    "topic": topic,
                    "template_family": prompt_family,
                    "class": sample_class,
                    "held_out_template": prompt_family in held_out_templates,
                    "behavior_label": int(sample_class in {"unsupported", "borderline"}),
                    "difficulty": round(0.45 + 0.12 * (sample_class == "borderline") + 0.05 * rng.random(), 2),
                    "prompt": prompt,
                    "question": question,
                    "evidence": evidence,
                    "expected_evidence_ids": expected_ids,
                    "generic_prompt": bool(split == "generic_unpaired"),
                }
            )
        rows[split] = split_rows
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    template_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    for row in rows:
        template_counts[row["template_family"]] = template_counts.get(row["template_family"], 0) + 1
        topic_counts[row["topic"]] = topic_counts.get(row["topic"], 0) + 1
    return {
        "size": len(rows),
        "behavior_positive_rate": mean(float(row["behavior_label"]) for row in rows),
        "template_counts": template_counts,
        "topic_counts": topic_counts,
    }
