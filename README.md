# eval_crosscoder

`eval_crosscoder` is a greenfield experiment framework that turns the proposal in
[proposal.md](C:/Users/pasca/Desktop/eval_crosscoder/proposal.md) into a runnable,
reproducible pipeline.

The repository ships with:

- a fixed seven-stage CLI:
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

## Quick start

Install the package in editable mode first, or set `PYTHONPATH=src`.

```powershell
python -m pip install -e .
python -m eval_crosscoder.cli prepare_data --config configs/pilot_smoke.json
python -m eval_crosscoder.cli train_lora --config configs/pilot_smoke.json --upstream-run <prepare_run>
python -m eval_crosscoder.cli cache_activations --config configs/pilot_smoke.json --upstream-run <train_lora_run>
python -m eval_crosscoder.cli train_methods --config configs/pilot_smoke.json --upstream-run <cache_run>
python -m eval_crosscoder.cli eval_predictive --config configs/pilot_smoke.json --upstream-run <methods_run>
python -m eval_crosscoder.cli eval_causal --config configs/pilot_smoke.json --upstream-run <predictive_run>
python -m eval_crosscoder.cli build_report --config configs/pilot_smoke.json --upstream-run <causal_run>
```

Hydra is now the preferred interface:

```powershell
python -m eval_crosscoder.cli prepare_data --config-name pilot_real
python -m eval_crosscoder.cli prepare_data --config-name pilot_real model=qwen2_5_1_5b_instruct
python -m eval_crosscoder.cli train_lora --config-name pilot_real --upstream-run <prepare_run> model=qwen2_5_1_5b_instruct lora.num_epochs=1
```

This keeps the command fixed while letting you switch models and hyperparameters
from the shell. The JSON configs are still accepted through `--config` for
backward compatibility.

For real experiments:

```powershell
python -m pip install -e .[real]
$env:PYTHONPATH='src'
python -m eval_crosscoder.cli prepare_data --config-name pilot_real model=qwen2_5_3b_instruct
python -m eval_crosscoder.cli train_lora --config-name pilot_real --upstream-run <prepare_run> model=qwen2_5_3b_instruct
python -m eval_crosscoder.cli cache_activations --config-name pilot_real --upstream-run <train_lora_run> model=qwen2_5_3b_instruct
python -m eval_crosscoder.cli train_methods --config-name pilot_real --upstream-run <cache_run> model=qwen2_5_3b_instruct
python -m eval_crosscoder.cli eval_predictive --config-name pilot_real --upstream-run <methods_run> model=qwen2_5_3b_instruct
python -m eval_crosscoder.cli eval_causal --config-name pilot_real --upstream-run <predictive_run> model=qwen2_5_3b_instruct
python -m eval_crosscoder.cli build_report --config-name pilot_real --upstream-run <causal_run> model=qwen2_5_3b_instruct
```

Every command creates a fresh directory under `runs/` and records its parent run,
so reports can reconstruct the lineage without mutating earlier artifacts.

## Configs

- Hydra experiment entrypoints:
  - [conf/pilot_smoke.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/pilot_smoke.yaml)
  - [conf/pilot_simulated.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/pilot_simulated.yaml)
  - [conf/pilot_real.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/pilot_real.yaml)
  - [conf/main_real.yaml](C:/Users/pasca/Desktop/eval_crosscoder/conf/main_real.yaml)
- Common override groups:
  - [conf/model](C:/Users/pasca/Desktop/eval_crosscoder/conf/model)
  - [conf/lora](C:/Users/pasca/Desktop/eval_crosscoder/conf/lora)
  - [conf/data](C:/Users/pasca/Desktop/eval_crosscoder/conf/data)
- [configs/pilot_smoke.json](C:/Users/pasca/Desktop/eval_crosscoder/configs/pilot_smoke.json)
  is the smallest end-to-end config for tests and local smoke runs.
- [configs/pilot_simulated.json](C:/Users/pasca/Desktop/eval_crosscoder/configs/pilot_simulated.json)
  mirrors the pilot phase in the proposal.
- [configs/main_simulated.json](C:/Users/pasca/Desktop/eval_crosscoder/configs/main_simulated.json)
  mirrors the main unsupported-citation abstention phase.
- [configs/pilot_real.json](C:/Users/pasca/Desktop/eval_crosscoder/configs/pilot_real.json)
  runs the pilot phase against a real Hugging Face model with LoRA fine-tuning.
- [configs/main_real.json](C:/Users/pasca/Desktop/eval_crosscoder/configs/main_real.json)
  runs the main unsupported-citation abstention experiment against a real model.

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
