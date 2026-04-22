from __future__ import annotations

import argparse
from pathlib import Path

from .cache.pipeline import build_activation_cache
from .causal.pipeline import eval_causal
from .config import compose_experiment_config, load_experiment_config
from .data.pipeline import prepare_data
from .eval.predictive import eval_predictive
from .lora.pipeline import train_lora
from .methods.factory import train_methods
from .reporting.pipeline import build_report
from .runs import add_artifact, create_run, mark_complete


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval-crosscoder")
    parser.add_argument("command", choices=_COMMANDS.keys())
    parser.add_argument("--config", help="Legacy JSON/YAML config file path.")
    parser.add_argument("--config-name", help="Hydra config name under --config-dir, without .yaml.")
    parser.add_argument("--config-dir", default="conf")
    parser.add_argument("--upstream-run")
    parser.add_argument("--workspace-root", default=".")
    args, hydra_overrides = parser.parse_known_args()

    workspace_root = Path(args.workspace_root).resolve()
    if args.config:
        config_path = Path(args.config).resolve()
        config = load_experiment_config(config_path)
    else:
        config_name = args.config_name or "pilot_smoke"
        config, config_path = compose_experiment_config(
            config_name=config_name,
            config_dir=args.config_dir,
            overrides=hydra_overrides,
        )
    stage_fn = _COMMANDS[args.command]
    run = create_run(
        workspace_root=workspace_root,
        config=config,
        config_path=config_path,
        stage=args.command,
        upstream_run=args.upstream_run,
    )
    resolved_config_path = run.write_json("config/resolved_config.json", config.to_dict())
    add_artifact(run, "resolved_config", resolved_config_path)
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
