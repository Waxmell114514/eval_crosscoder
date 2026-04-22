from __future__ import annotations

import argparse
from pathlib import Path

from .cache.pipeline import build_activation_cache
from .causal.pipeline import eval_causal
from .config import load_experiment_config
from .data.pipeline import prepare_data
from .eval.predictive import eval_predictive
from .lora.pipeline import train_lora
from .methods.factory import train_methods
from .reporting.pipeline import build_report
from .runs import create_run, mark_complete


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval-crosscoder")
    parser.add_argument("command", choices=_COMMANDS.keys())
    parser.add_argument("--config", required=True)
    parser.add_argument("--upstream-run")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    config = load_experiment_config(config_path)
    stage_fn = _COMMANDS[args.command]
    run = create_run(
        workspace_root=workspace_root,
        config=config,
        config_path=config_path,
        stage=args.command,
        upstream_run=args.upstream_run,
    )
    result = stage_fn(config, args.upstream_run, run)
    mark_complete(run, extra={"result_keys": sorted(result.keys()) if isinstance(result, dict) else []})
    print(run.path)


def _prepare(config, _upstream_run, run):
    return prepare_data(config, run)


_COMMANDS = {
    "prepare_data": _prepare,
    "train_lora": train_lora,
    "cache_activations": build_activation_cache,
    "train_methods": train_methods,
    "eval_predictive": eval_predictive,
    "eval_causal": eval_causal,
    "build_report": build_report,
}


if __name__ == "__main__":
    main()
