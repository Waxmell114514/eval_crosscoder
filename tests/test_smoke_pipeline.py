from __future__ import annotations

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_crosscoder.cli import _prepare, run_pipeline
from eval_crosscoder.config import compose_experiment_config
from eval_crosscoder.data.pipeline import load_split
from eval_crosscoder.runs import create_run, mark_complete


class SmokePipelineTest(unittest.TestCase):
    def test_pilot_smoke_pipeline(self) -> None:
        config, config_path = compose_experiment_config(
            config_name="pilot_smoke",
            config_dir=REPO_ROOT / "conf",
        )
        final_run, report_path = run_pipeline(
            workspace_root=REPO_ROOT,
            config=config,
            config_path=config_path,
        )
        self.assertIsNotNone(report_path)
        assert report_path is not None
        self.assertTrue(report_path.exists(), "final report should be generated")
        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("Predictive Ranking", report_text)
        self.assertIn("Causal Ranking", report_text)
        self.assertTrue(final_run.exists())

    def test_hydra_override_can_switch_real_model(self) -> None:
        config, config_path = compose_experiment_config(
            config_name="pilot_real",
            config_dir=REPO_ROOT / "conf",
            overrides=["model=qwen2_5_1_5b_instruct", "lora.num_epochs=1"],
        )
        self.assertEqual(config_path.name, "pilot_real.yaml")
        self.assertEqual(config.backend, "huggingface")
        self.assertEqual(config.model["base_model_name_or_path"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(config.lora["num_epochs"], 1)

    def test_hydra_config_prepare_stage_is_usable(self) -> None:
        config, config_path = compose_experiment_config(
            config_name="pilot_real",
            config_dir=REPO_ROOT / "conf",
            overrides=["model=qwen2_5_1_5b_instruct"],
        )
        run = create_run(REPO_ROOT, config, config_path, "prepare_data", None)
        result = _prepare(config, None, run)
        mark_complete(run, extra={"result_keys": sorted(result.keys()) if isinstance(result, dict) else []})
        rows = load_split(run.path, "train")
        self.assertTrue(rows)
        self.assertIn("target_text", rows[0])


if __name__ == "__main__":
    unittest.main()
