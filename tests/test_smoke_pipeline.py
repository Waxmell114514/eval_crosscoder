from __future__ import annotations

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_crosscoder.cache.pipeline import build_activation_cache
from eval_crosscoder.causal.pipeline import eval_causal
from eval_crosscoder.cli import _prepare
from eval_crosscoder.config import load_experiment_config
from eval_crosscoder.data.pipeline import load_split
from eval_crosscoder.eval.predictive import eval_predictive
from eval_crosscoder.lora.pipeline import train_lora
from eval_crosscoder.methods.factory import train_methods
from eval_crosscoder.reporting.pipeline import build_report
from eval_crosscoder.runs import create_run, mark_complete


class SmokePipelineTest(unittest.TestCase):
    def test_pilot_smoke_pipeline(self) -> None:
        config_path = REPO_ROOT / "configs" / "pilot_smoke.json"
        config = load_experiment_config(config_path)
        workspace_root = REPO_ROOT
        upstream = None
        stages = [
            ("prepare_data", _prepare),
            ("train_lora", train_lora),
            ("cache_activations", build_activation_cache),
            ("train_methods", train_methods),
            ("eval_predictive", eval_predictive),
            ("eval_causal", eval_causal),
            ("build_report", build_report),
        ]
        final_run = None
        for stage_name, stage_fn in stages:
            run = create_run(workspace_root, config, config_path, stage_name, upstream)
            result = stage_fn(config, upstream, run)
            mark_complete(run, extra={"result_keys": sorted(result.keys()) if isinstance(result, dict) else []})
            upstream = str(run.path)
            final_run = run.path
        assert final_run is not None
        report_path = final_run / "report" / "report.md"
        self.assertTrue(report_path.exists(), "final report should be generated")
        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("Predictive Ranking", report_text)
        self.assertIn("Causal Ranking", report_text)

    def test_real_config_prepare_stage_is_usable_without_real_deps(self) -> None:
        config_path = REPO_ROOT / "configs" / "pilot_real.json"
        config = load_experiment_config(config_path)
        run = create_run(REPO_ROOT, config, config_path, "prepare_data", None)
        result = _prepare(config, None, run)
        mark_complete(run, extra={"result_keys": sorted(result.keys()) if isinstance(result, dict) else []})
        rows = load_split(run.path, "train")
        self.assertTrue(rows, "real config prepare_data should create train rows")
        self.assertIn("target_text", rows[0])
        self.assertEqual(config.backend, "huggingface")


if __name__ == "__main__":
    unittest.main()
