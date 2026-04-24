# eval_crosscoder

`eval_crosscoder` is a greenfield experiment framework that turns the proposal in
[proposal.md](C:/Users/pasca/Desktop/eval_crosscoder/proposal.md) into a runnable,
reproducible pipeline.

The repository ships with:

- a fixed seven-stage experimental pipeline:
  - `prepare_data`
  - `train_lora`
  - `cache_activations`
  - `train_methods`
  - `eval_predictive`
  - `eval_causal`
  - `build_report`
- proposal-aligned public interfaces:
  - `TaskSpec`
  - `CacheSpec`
  - `MethodResult`
  - `EvalSummary`
  - `CausalSummary`
- a fully offline `simulated` backend that behaves like a controlled model organism
  so the whole pipeline can be tested without downloading model weights.
- a real `huggingface` backend that trains PEFT LoRA adapters, caches residual
  activations from actual model hidden states, and runs causal steering/ablation
  directly on transformer layers.
- a Hydra config tree under [conf](C:/Users/pasca/Desktop/eval_crosscoder/conf)
  so models, tasks, LoRA settings, and evaluation knobs can be swapped by override
  instead of editing large JSON files.

The simulated backend is intentionally explicit about its limits: it validates the
protocol, metrics, artifacts, and stage boundaries. For real experiments, use the
`huggingface` backend configs and install the optional real dependencies.

On Python 3.13, the repository uses an OmegaConf-backed Hydra-compatible fallback
composer by default because `hydra-core` is not reliably importable there yet.
The CLI surface and config layout stay Hydra-style, so you can still use
`--config-name` and dotted overrides normally.

## Quick start

Install the package in editable mode first, or set `PYTHONPATH=src`.

```powershell
python -m pip install -e .
python -m eval_crosscoder.cli run_pipeline --config-name pilot_smoke
```

The one-shot entrypoint runs all seven stages in order, prints each run directory as it
goes, then prints the final report path followed by the final run directory on the last
two lines.

Hydra-style composition is now the only experiment config interface:

```powershell
python -m eval_crosscoder.cli run_pipeline --config-name pilot_real
python -m eval_crosscoder.cli run_pipeline --config-name pilot_real model=qwen2_5_1_5b_instruct
python -m eval_crosscoder.cli run_pipeline --config-name pilot_real model=qwen2_5_1_5b_instruct lora.num_epochs=1
```

For real experiments:

```powershell
python -m pip install -e .[real]
$env:PYTHONPATH='src'
python -m eval_crosscoder.cli run_pipeline --config-name pilot_real model=qwen2_5_3b_instruct
```

For an `RTX 5070 Laptop` class machine with `8GB` VRAM, use the local presets:

```powershell
python -m eval_crosscoder.cli run_pipeline --config-name pilot_real_rtx5070_laptop_smoke
python -m eval_crosscoder.cli run_pipeline --config-name pilot_real_rtx5070_laptop
```

For Kaggle `T4 x2`, start with the T4 preset. The current backend uses one GPU
(`cuda:0`) and does not yet shard across both T4s:

```bash
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

python -u -m eval_crosscoder.cli run_pipeline --config-name pilot_real_kaggle_t4_smoke
python -u -m eval_crosscoder.cli run_pipeline --config-name pilot_real_kaggle_t4
```

If you want to actually use both T4s with the current codebase, use the dual-GPU
model-sharding presets:

```bash
python -u -m eval_crosscoder.cli run_pipeline --config-name pilot_real_kaggle_t4x2_smoke
python -u -m eval_crosscoder.cli run_pipeline --config-name pilot_real_kaggle_t4x2
```

These use `model.device_map=balanced`, so the model is spread across both GPUs.
This is the lowest-friction way to use both cards today, but it is model parallelism,
not true multi-process data parallel training, so speedups are helpful rather than
near-linear.

If you need to resume from a previously completed stage, `run_pipeline` also accepts:

```powershell
python -m eval_crosscoder.cli run_pipeline --config-name pilot_real_rtx5070_laptop --from-stage train_methods --upstream-run <cache_run>
python -m eval_crosscoder.cli run_pipeline --config-name pilot_real_rtx5070_laptop --from-stage eval_causal --to-stage build_report --upstream-run <predictive_run>
```

Every command creates a fresh directory under `runs/` and records its parent run,
so reports can reconstruct the lineage without mutating earlier artifacts.

## Configs

- Hydra experiment entrypoints:
  - [conf/pilot_smoke.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/pilot_smoke.yaml)
  - [conf/pilot_simulated.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/pilot_simulated.yaml)
  - [conf/pilot_real.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/pilot_real.yaml)
  - [conf/pilot_real_rtx5070_laptop_smoke.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/pilot_real_rtx5070_laptop_smoke.yaml)
  - [conf/pilot_real_rtx5070_laptop.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/pilot_real_rtx5070_laptop.yaml)
  - [conf/main_real.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/main_real.yaml)
  - [conf/main_real_rtx5070_laptop.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/main_real_rtx5070_laptop.yaml)
- Common override groups:
  - [conf/model](C:/Users/pasca/Desktop/eval_crosscoder/conf/model)
  - [conf/lora](C:/Users/pasca/Desktop/eval_crosscoder/conf/lora)
  - [conf/data](C:/Users/pasca/Desktop/eval_crosscoder/conf/data)

## Notes

- The `simulated` backend uses deterministic prompt hashing plus a learned adapter
  artifact to synthesize:
  - matched base/LoRA activations
  - held-out behavior deltas
  - recoverable target latents and off-target contamination
- The `huggingface` backend expects:
  - `model.base_model_name_or_path`
  - `model.tokenizer_name_or_path` (optional, defaults to the base model)
  - `model.max_seq_length`, `model.max_new_tokens`, `model.device`, `model.torch_dtype`
  - `lora.num_epochs`, `lora.batch_size`, `lora.gradient_accumulation_steps`, `lora.learning_rate`
  - optional local data via `data.source = "local_jsonl"` and `data.split_paths`
- `pilot_real` and `main_real` default to `Qwen/Qwen2.5-3B-Instruct` in Hydra so
  you can run without Gemma access. If you later get access, switch with
  `model=gemma2_2b_it`.
- `train_methods` supports the proposal's comparison set:
  - `raw_diff`
  - `mean_diff`
  - `pca`
  - `behavior_probe`
  - `standard_crosscoder`
  - `batch_topk_crosscoder`
  - `dfc`
  - `delta_crosscoder`
  - `single_model_sae`
- `build_report` emits a Markdown summary that includes gate checks, rankings, and
  acceptance status against the thresholds in the plan.
- Real causal interventions are implemented as forward hooks on the chosen
  transformer layer, applied during generation on the final token state.
