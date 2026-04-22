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

The simulated backend is intentionally explicit about its limits: it validates the
protocol, metrics, artifacts, and stage boundaries, but it is not a substitute for
running the same pipeline against real `Gemma 2 2B-it` or `Qwen2.5-3B-Instruct`
weights.

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

Every command creates a fresh directory under `runs/` and records its parent run,
so reports can reconstruct the lineage without mutating earlier artifacts.

## Configs

- [configs/pilot_smoke.json](C:/Users/pasca/Desktop/eval_crosscoder/configs/pilot_smoke.json)
  is the smallest end-to-end config for tests and local smoke runs.
- [configs/pilot_simulated.json](C:/Users/pasca/Desktop/eval_crosscoder/configs/pilot_simulated.json)
  mirrors the pilot phase in the proposal.
- [configs/main_simulated.json](C:/Users/pasca/Desktop/eval_crosscoder/configs/main_simulated.json)
  mirrors the main unsupported-citation abstention phase.

## Notes

- The `simulated` backend uses deterministic prompt hashing plus a learned adapter
  artifact to synthesize:
  - matched base/LoRA activations
  - held-out behavior deltas
  - recoverable target latents and off-target contamination
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
