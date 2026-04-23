from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .cache.pipeline import build_activation_cache
from .causal.pipeline import eval_causal
from .config import compose_experiment_config
from .data.pipeline import prepare_data
from .eval.predictive import eval_predictive
from .lora.pipeline import train_lora
from .methods.factory import train_methods
from .reporting.pipeline import build_report
from .runs import add_artifact, create_run, mark_complete


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval-crosscoder")
    parser.add_argument("command", choices=[*_PIPELINE_STAGES, "run_pipeline"])
    parser.add_argument("--config-name", default="pilot_smoke", help="Hydra config name under --config-dir, without .yaml.")
    parser.add_argument("--config-dir", default="conf")
    parser.add_argument("--upstream-run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--from-stage", choices=_PIPELINE_STAGES, default=_PIPELINE_STAGES[0])
    parser.add_argument("--to-stage", choices=_PIPELINE_STAGES, default=_PIPELINE_STAGES[-1])
    args, hydra_overrides = parser.parse_known_args()

    workspace_root = Path(args.workspace_root).resolve()
    config, config_path = compose_experiment_config(
        config_name=args.config_name,
        config_dir=args.config_dir,
        overrides=hydra_overrides,
    )
    if args.command == "run_pipeline":
        final_run, report_path = run_pipeline(
            workspace_root=workspace_root,
            config=config,
            config_path=config_path,
            upstream_run=args.upstream_run,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
        )
        if report_path is not None:
            print(report_path)
        print(final_run)
        return
    if args.from_stage != _PIPELINE_STAGES[0] or args.to_stage != _PIPELINE_STAGES[-1]:
        parser.error("--from-stage and --to-stage are only valid with the run_pipeline command.")
    run = run_stage(
        stage=args.command,
        workspace_root=workspace_root,
        config=config,
        config_path=config_path,
        upstream_run=args.upstream_run,
    )
    print(run.path)


def _prepare(config, _upstream_run, run):
    return prepare_data(config, run)


def run_stage(
    stage: str,
    workspace_root: Path,
    config: Any,
    config_path: Path,
    upstream_run: str | None,
):
    stage_fn = _STAGE_COMMANDS[stage]
    run = create_run(
        workspace_root=workspace_root,
        config=config,
        config_path=config_path,
        stage=stage,
        upstream_run=upstream_run,
    )
    resolved_config_path = run.write_json("config/resolved_config.json", config.to_dict())
    add_artifact(run, "resolved_config", resolved_config_path)
    result = stage_fn(config, upstream_run, run)
    mark_complete(run, extra={"result_keys": sorted(result.keys()) if isinstance(result, dict) else []})
    return run


def run_pipeline(
    workspace_root: Path,
    config: Any,
    config_path: Path,
    upstream_run: str | None = None,
    from_stage: str = "prepare_data",
    to_stage: str = "build_report",
) -> tuple[Path, Path | None]:
    stages = _pipeline_slice(from_stage, to_stage)
    current_upstream = upstream_run
    if stages[0] == "prepare_data":
        current_upstream = None
    elif current_upstream is None:
        raise ValueError(f"run_pipeline starting from {stages[0]!r} requires --upstream-run.")
    final_run = None
    for index, stage in enumerate(stages, start=1):
        print(f"[{index}/{len(stages)}] {stage}")
        run = run_stage(
            stage=stage,
            workspace_root=workspace_root,
            config=config,
            config_path=config_path,
            upstream_run=current_upstream,
        )
        print(f"  -> {run.path}")
        current_upstream = str(run.path)
        final_run = run.path
    assert final_run is not None
    report_path = final_run / "report" / "report.md" if stages[-1] == "build_report" else None
    return final_run, report_path


def _pipeline_slice(from_stage: str, to_stage: str) -> list[str]:
    start_index = _PIPELINE_STAGES.index(from_stage)
    end_index = _PIPELINE_STAGES.index(to_stage)
    if start_index > end_index:
        raise ValueError(f"--from-stage {from_stage!r} must come before --to-stage {to_stage!r}.")
    return _PIPELINE_STAGES[start_index : end_index + 1]


_STAGE_COMMANDS = {
    "prepare_data": _prepare,
    "train_lora": train_lora,
    "cache_activations": build_activation_cache,
    "train_methods": train_methods,
    "eval_predictive": eval_predictive,
    "eval_causal": eval_causal,
    "build_report": build_report,
}
_PIPELINE_STAGES = list(_STAGE_COMMANDS.keys())


if __name__ == "__main__":
    main()
