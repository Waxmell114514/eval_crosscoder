from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from ..cache.pipeline import load_cache_bundle
from ..config import ExperimentConfig
from ..runs import RunContext, add_artifact, lineage_from_run, load_manifest
from ..specs import MethodResult
from ..utils import normalize


EPS = 1e-8


@dataclass(slots=True)
class ProjectionMethodState:
    name: str
    layer: int
    basis: np.ndarray
    latent_ranking: list[int]
    input_mode: str = "delta"
    decoder_stats: dict[str, float] = field(default_factory=dict)
    exclusivity_stats: dict[str, float] = field(default_factory=dict)
    top_examples: list[dict[str, Any]] = field(default_factory=list)

    def pair_features(self, base: np.ndarray, lora: np.ndarray) -> np.ndarray:
        source = _pair_source(base, lora, self.input_mode)
        feats = source @ self.basis.T
        return feats[:, self.latent_ranking]

    def single_features(self, activations: np.ndarray, side: str) -> np.ndarray:
        sign = -1.0 if side == "base" and self.input_mode == "delta" else 1.0
        feats = sign * (activations @ self.basis.T)
        return feats[:, self.latent_ranking]

    def intervention_vector(self, top_k: int) -> np.ndarray:
        count = min(top_k, len(self.latent_ranking))
        if count <= 0:
            return np.zeros(self.basis.shape[1], dtype=np.float32)
        return normalize(self.basis[self.latent_ranking[:count]].sum(axis=0))

    def to_result(self) -> MethodResult:
        return MethodResult(
            name=self.name,
            layer=self.layer,
            latent_ids=self.latent_ranking,
            decoder_stats=self.decoder_stats,
            exclusivity_stats=self.exclusivity_stats,
            top_examples=self.top_examples,
        )


@dataclass(slots=True)
class AutoencoderMethodState:
    name: str
    layer: int
    input_mode: str
    encoder_weight: np.ndarray
    encoder_bias: np.ndarray
    intervention_basis: np.ndarray
    latent_ranking: list[int]
    topk: int = 0
    decoder_stats: dict[str, float] = field(default_factory=dict)
    exclusivity_stats: dict[str, float] = field(default_factory=dict)
    top_examples: list[dict[str, Any]] = field(default_factory=list)

    def pair_features(self, base: np.ndarray, lora: np.ndarray) -> np.ndarray:
        source = _pair_source(base, lora, self.input_mode)
        latent = _encode_relu(source, self.encoder_weight, self.encoder_bias, self.topk)
        return latent[:, self.latent_ranking]

    def single_features(self, activations: np.ndarray, side: str) -> np.ndarray:
        feats = activations @ self.intervention_basis.T
        if side == "base":
            feats = -feats
        return feats[:, self.latent_ranking]

    def intervention_vector(self, top_k: int) -> np.ndarray:
        count = min(top_k, len(self.latent_ranking))
        if count <= 0:
            return np.zeros(self.intervention_basis.shape[1], dtype=np.float32)
        return normalize(self.intervention_basis[self.latent_ranking[:count]].sum(axis=0))

    def to_result(self) -> MethodResult:
        return MethodResult(
            name=self.name,
            layer=self.layer,
            latent_ids=self.latent_ranking,
            decoder_stats=self.decoder_stats,
            exclusivity_stats=self.exclusivity_stats,
            top_examples=self.top_examples,
        )


def train_methods(config: ExperimentConfig, upstream_run: str | Path, run: RunContext) -> dict[str, Any]:
    cache_manifest = load_manifest(upstream_run)
    line = lineage_from_run(upstream_run)
    behavior_eval = _load_behavior_eval_from_lineage(line)
    enabled_methods = list(config.methods.get("enabled", []))
    if behavior_eval and behavior_eval.get("phase_gate_passed"):
        enabled_methods.extend(config.methods.get("post_gate_methods", []))
    enabled_methods.extend(config.methods.get("always_include_methods", []))
    enabled_methods = list(dict.fromkeys(enabled_methods))

    cache_summary_path = cache_manifest["artifacts"]["cache_summary"]
    cache_summary = _read_json(Path(cache_summary_path))
    layer_ids = [int(layer) for layer in cache_summary["layers"]]
    methods_dir = run.artifact("methods")
    methods_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"enabled_methods": enabled_methods, "layers": {}}
    for layer in layer_ids:
        train_bundle = load_cache_bundle(upstream_run, "train", layer)
        val_bundle = load_cache_bundle(upstream_run, "val", layer)
        layer_summary: dict[str, Any] = {}
        for method_name in enabled_methods:
            state = _fit_method(method_name, layer, train_bundle, val_bundle, config)
            artifact_path = methods_dir / f"{method_name}_layer{layer}.pkl"
            with artifact_path.open("wb") as handle:
                pickle.dump(state, handle)
            add_artifact(run, f"method_{method_name}_layer{layer}", artifact_path)
            result = state.to_result()
            result.artifact_path = str(artifact_path.resolve())
            layer_summary[method_name] = result.to_dict()
        summary["layers"][str(layer)] = layer_summary
    summary_path = run.write_json("methods/summary.json", summary)
    add_artifact(run, "methods_summary", summary_path)
    return summary


def load_method(run_path: str | Path, method_name: str, layer: int) -> ProjectionMethodState | AutoencoderMethodState:
    method_path = Path(run_path) / "methods" / f"{method_name}_layer{layer}.pkl"
    with method_path.open("rb") as handle:
        return pickle.load(handle)


def _fit_method(
    method_name: str,
    layer: int,
    train_bundle: dict[str, np.ndarray],
    val_bundle: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> ProjectionMethodState | AutoencoderMethodState:
    training_cfg = config.methods.get("training", {})
    latent_dim = int(training_cfg.get("latent_dim", 16))
    if method_name == "raw_diff":
        return _fit_raw_diff(layer, train_bundle, latent_dim)
    if method_name == "mean_diff":
        return _fit_mean_diff(layer, train_bundle)
    if method_name == "pca":
        return _fit_pca(layer, train_bundle, latent_dim)
    if method_name == "behavior_probe":
        return _fit_behavior_probe(layer, train_bundle)
    if method_name == "single_model_sae":
        return _fit_single_model_sae(layer, train_bundle, training_cfg)
    if method_name == "standard_crosscoder":
        return _fit_pair_autoencoder(layer, train_bundle, training_cfg, method_name="standard_crosscoder")
    if method_name == "batch_topk_crosscoder":
        return _fit_pair_autoencoder(layer, train_bundle, training_cfg, method_name="batch_topk_crosscoder")
    if method_name == "dfc":
        return _fit_dfc(layer, train_bundle, training_cfg)
    if method_name == "delta_crosscoder":
        return _fit_delta_crosscoder(layer, train_bundle, training_cfg)
    raise ValueError(f"Unknown method: {method_name}")


def _fit_raw_diff(layer: int, bundle: dict[str, np.ndarray], latent_dim: int) -> ProjectionMethodState:
    mean_delta = bundle["delta"].mean(axis=0)
    ranking = np.argsort(-np.abs(mean_delta))[:latent_dim]
    basis = np.eye(bundle["delta"].shape[1], dtype=np.float32)[ranking]
    full_ranking = list(range(len(ranking)))
    feats_base = bundle["base"][:, ranking]
    feats_lora = bundle["lora"][:, ranking]
    return ProjectionMethodState(
        name="raw_diff",
        layer=layer,
        basis=basis,
        latent_ranking=full_ranking,
        input_mode="delta",
        decoder_stats={"mean_abs_delta": float(np.abs(mean_delta[ranking]).mean())},
        exclusivity_stats=_exclusivity_stats(feats_base, feats_lora),
        top_examples=_top_examples(bundle, bundle["delta"][:, ranking], full_ranking),
    )


def _fit_mean_diff(layer: int, bundle: dict[str, np.ndarray]) -> ProjectionMethodState:
    mean_delta = normalize(bundle["delta"].mean(axis=0)).astype(np.float32)
    basis = mean_delta[None, :]
    feats_base = bundle["base"] @ basis.T
    feats_lora = bundle["lora"] @ basis.T
    return ProjectionMethodState(
        name="mean_diff",
        layer=layer,
        basis=basis,
        latent_ranking=[0],
        input_mode="delta",
        decoder_stats={"vector_norm": float(np.linalg.norm(mean_delta))},
        exclusivity_stats=_exclusivity_stats(feats_base, feats_lora),
        top_examples=_top_examples(bundle, bundle["delta"] @ basis.T, [0]),
    )


def _fit_pca(layer: int, bundle: dict[str, np.ndarray], latent_dim: int) -> ProjectionMethodState:
    n_components = min(latent_dim, bundle["delta"].shape[0], bundle["delta"].shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(bundle["delta"])
    basis = pca.components_.astype(np.float32)
    scores = bundle["delta"] @ basis.T
    feats_base = bundle["base"] @ basis.T
    feats_lora = bundle["lora"] @ basis.T
    return ProjectionMethodState(
        name="pca",
        layer=layer,
        basis=basis,
        latent_ranking=list(np.argsort(-pca.explained_variance_ratio_)),
        input_mode="delta",
        decoder_stats={"explained_variance": float(pca.explained_variance_ratio_.sum())},
        exclusivity_stats=_exclusivity_stats(feats_base, feats_lora),
        top_examples=_top_examples(bundle, scores, list(range(scores.shape[1]))),
    )


def _fit_behavior_probe(layer: int, bundle: dict[str, np.ndarray]) -> ProjectionMethodState:
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(bundle["delta"], bundle["behavior_label"])
    coef = model.coef_[0].astype(np.float32)
    basis = normalize(coef)[None, :]
    feats_base = bundle["base"] @ basis.T
    feats_lora = bundle["lora"] @ basis.T
    return ProjectionMethodState(
        name="behavior_probe",
        layer=layer,
        basis=basis,
        latent_ranking=[0],
        input_mode="delta",
        decoder_stats={"coef_norm": float(np.linalg.norm(coef))},
        exclusivity_stats=_exclusivity_stats(feats_base, feats_lora),
        top_examples=_top_examples(bundle, bundle["delta"] @ basis.T, [0]),
    )


def _fit_single_model_sae(
    layer: int,
    bundle: dict[str, np.ndarray],
    training_cfg: dict[str, Any],
) -> AutoencoderMethodState:
    latent_dim = int(training_cfg.get("latent_dim", 16))
    state = _train_standard_autoencoder(
        bundle["lora"],
        latent_dim=latent_dim,
        epochs=int(training_cfg.get("epochs", 30)),
        lr=float(training_cfg.get("lr", 1e-2)),
        sparsity=float(training_cfg.get("sparsity_weight", 1e-3)),
        topk=0,
    )
    pair_feats = _encode_relu(bundle["lora"], state["encoder_weight"], state["encoder_bias"], 0) - _encode_relu(
        bundle["base"], state["encoder_weight"], state["encoder_bias"], 0
    )
    ranking = _rank_latents(pair_feats, bundle["behavior_label"])
    intervention_basis = state["decoder_weight"]
    return AutoencoderMethodState(
        name="single_model_sae",
        layer=layer,
        input_mode="lora_only",
        encoder_weight=state["encoder_weight"],
        encoder_bias=state["encoder_bias"],
        intervention_basis=intervention_basis,
        latent_ranking=ranking,
        decoder_stats={"decoder_norm_mean": float(np.linalg.norm(intervention_basis, axis=1).mean())},
        exclusivity_stats=_exclusivity_stats(bundle["base"] @ intervention_basis.T, bundle["lora"] @ intervention_basis.T),
        top_examples=_top_examples(bundle, pair_feats, ranking),
    )


def _fit_pair_autoencoder(
    layer: int,
    bundle: dict[str, np.ndarray],
    training_cfg: dict[str, Any],
    method_name: str,
) -> AutoencoderMethodState:
    latent_dim = int(training_cfg.get("latent_dim", 16))
    topk = int(training_cfg.get("batch_topk", 4)) if method_name == "batch_topk_crosscoder" else 0
    pair_input = np.concatenate([bundle["base"], bundle["lora"]], axis=1)
    state = _train_standard_autoencoder(
        pair_input,
        latent_dim=latent_dim,
        epochs=int(training_cfg.get("epochs", 30)),
        lr=float(training_cfg.get("lr", 1e-2)),
        sparsity=float(training_cfg.get("sparsity_weight", 1e-3)),
        topk=topk,
    )
    pair_feats = _encode_relu(pair_input, state["encoder_weight"], state["encoder_bias"], topk)
    ranking = _rank_latents(pair_feats, bundle["behavior_label"])
    hidden_dim = bundle["base"].shape[1]
    decoder = state["decoder_weight"]
    intervention_basis = decoder[:, hidden_dim:] - decoder[:, :hidden_dim]
    return AutoencoderMethodState(
        name=method_name,
        layer=layer,
        input_mode="pair_concat",
        encoder_weight=state["encoder_weight"],
        encoder_bias=state["encoder_bias"],
        intervention_basis=intervention_basis,
        latent_ranking=ranking,
        topk=topk,
        decoder_stats={"decoder_norm_mean": float(np.linalg.norm(intervention_basis, axis=1).mean())},
        exclusivity_stats=_exclusivity_stats(bundle["base"] @ intervention_basis.T, bundle["lora"] @ intervention_basis.T),
        top_examples=_top_examples(bundle, pair_feats, ranking),
    )


def _fit_dfc(layer: int, bundle: dict[str, np.ndarray], training_cfg: dict[str, Any]) -> AutoencoderMethodState:
    latent_dim = int(training_cfg.get("latent_dim", 18))
    shared_fraction = float(training_cfg.get("shared_fraction", 0.33))
    hidden_dim = bundle["base"].shape[1]
    input_data = np.concatenate([bundle["base"], bundle["lora"]], axis=1)
    state = _train_dfc_autoencoder(
        input_data,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        shared_fraction=shared_fraction,
        epochs=int(training_cfg.get("epochs", 30)),
        lr=float(training_cfg.get("lr", 1e-2)),
        sparsity=float(training_cfg.get("sparsity_weight", 1e-3)),
    )
    pair_feats = _encode_relu(input_data, state["encoder_weight"], state["encoder_bias"], 0)
    ranking = _rank_latents(pair_feats, bundle["behavior_label"])
    intervention_basis = state["intervention_basis"]
    return AutoencoderMethodState(
        name="dfc",
        layer=layer,
        input_mode="pair_concat",
        encoder_weight=state["encoder_weight"],
        encoder_bias=state["encoder_bias"],
        intervention_basis=intervention_basis,
        latent_ranking=ranking,
        topk=0,
        decoder_stats={"shared_fraction": shared_fraction},
        exclusivity_stats=_exclusivity_stats(bundle["base"] @ intervention_basis.T, bundle["lora"] @ intervention_basis.T),
        top_examples=_top_examples(bundle, pair_feats, ranking),
    )


def _fit_delta_crosscoder(
    layer: int,
    bundle: dict[str, np.ndarray],
    training_cfg: dict[str, Any],
) -> AutoencoderMethodState:
    latent_dim = int(training_cfg.get("latent_dim", 18))
    hidden_dim = bundle["base"].shape[1]
    mean_pair = (bundle["base"] + bundle["lora"]) / 2.0
    delta_input = np.concatenate([bundle["delta"], mean_pair], axis=1)
    state = _train_delta_autoencoder(
        delta_input,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        epochs=int(training_cfg.get("epochs", 30)),
        lr=float(training_cfg.get("lr", 1e-2)),
        sparsity=float(training_cfg.get("sparsity_weight", 1e-3)),
    )
    pair_feats = _encode_relu(delta_input, state["encoder_weight"], state["encoder_bias"], 0)
    ranking = _rank_latents(pair_feats, bundle["behavior_label"])
    intervention_basis = state["intervention_basis"]
    return AutoencoderMethodState(
        name="delta_crosscoder",
        layer=layer,
        input_mode="delta_mean",
        encoder_weight=state["encoder_weight"],
        encoder_bias=state["encoder_bias"],
        intervention_basis=intervention_basis,
        latent_ranking=ranking,
        topk=0,
        decoder_stats={"delta_decoder_norm": float(np.linalg.norm(intervention_basis, axis=1).mean())},
        exclusivity_stats=_exclusivity_stats(bundle["base"] @ intervention_basis.T, bundle["lora"] @ intervention_basis.T),
        top_examples=_top_examples(bundle, pair_feats, ranking),
    )


def _train_standard_autoencoder(
    features: np.ndarray,
    latent_dim: int,
    epochs: int,
    lr: float,
    sparsity: float,
    topk: int,
) -> dict[str, np.ndarray]:
    x = torch.tensor(features, dtype=torch.float32)
    model = _StandardAE(input_dim=features.shape[1], latent_dim=latent_dim, topk=topk)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        recon, latent = model(x)
        loss = ((recon - x) ** 2).mean() + sparsity * latent.abs().mean()
        loss.backward()
        opt.step()
    return {
        "encoder_weight": model.encoder.weight.detach().cpu().numpy().T.astype(np.float32),
        "encoder_bias": model.encoder.bias.detach().cpu().numpy().astype(np.float32),
        "decoder_weight": model.decoder.weight.detach().cpu().numpy().T.astype(np.float32),
    }


def _train_dfc_autoencoder(
    features: np.ndarray,
    hidden_dim: int,
    latent_dim: int,
    shared_fraction: float,
    epochs: int,
    lr: float,
    sparsity: float,
) -> dict[str, np.ndarray]:
    x = torch.tensor(features, dtype=torch.float32)
    shared_dim = max(2, int(round(latent_dim * shared_fraction)))
    exclusive_dim = max(2, (latent_dim - shared_dim) // 2)
    latent_dim = exclusive_dim * 2 + shared_dim
    model = _DFCAE(hidden_dim=hidden_dim, latent_dim=latent_dim, shared_dim=shared_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        recon, latent = model(x)
        loss = ((recon - x) ** 2).mean() + sparsity * latent.abs().mean()
        loss.backward()
        opt.step()
    basis = model.intervention_basis().detach().cpu().numpy().astype(np.float32)
    return {
        "encoder_weight": model.encoder.weight.detach().cpu().numpy().T.astype(np.float32),
        "encoder_bias": model.encoder.bias.detach().cpu().numpy().astype(np.float32),
        "intervention_basis": basis,
    }


def _train_delta_autoencoder(
    features: np.ndarray,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    lr: float,
    sparsity: float,
) -> dict[str, np.ndarray]:
    x = torch.tensor(features, dtype=torch.float32)
    model = _DeltaAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        recon, latent = model(x)
        loss = ((recon - x) ** 2).mean() + sparsity * latent.abs().mean()
        loss.backward()
        opt.step()
    return {
        "encoder_weight": model.encoder.weight.detach().cpu().numpy().T.astype(np.float32),
        "encoder_bias": model.encoder.bias.detach().cpu().numpy().astype(np.float32),
        "intervention_basis": model.intervention_basis().detach().cpu().numpy().astype(np.float32),
    }


class _StandardAE(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, topk: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, input_dim, bias=False)
        self.topk = topk

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = torch.relu(self.encoder(x))
        if self.topk > 0:
            values, indices = torch.topk(latent, k=min(self.topk, latent.shape[1]), dim=1)
            mask = torch.zeros_like(latent)
            mask.scatter_(1, indices, 1.0)
            latent = latent * mask
        return latent

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        return self.decoder(latent), latent


class _DFCAE(torch.nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, shared_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared_dim = shared_dim
        exclusive_dim = (latent_dim - shared_dim) // 2
        self.base_exclusive_dim = exclusive_dim
        self.lora_exclusive_dim = exclusive_dim
        self.encoder = torch.nn.Linear(hidden_dim * 2, latent_dim)
        self.base_decoder = torch.nn.Linear(self.base_exclusive_dim + shared_dim, hidden_dim, bias=False)
        self.lora_decoder = torch.nn.Linear(shared_dim + self.lora_exclusive_dim, hidden_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def split(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = latent[:, : self.base_exclusive_dim]
        s = latent[:, self.base_exclusive_dim : self.base_exclusive_dim + self.shared_dim]
        l = latent[:, self.base_exclusive_dim + self.shared_dim :]
        return b, s, l

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        b, s, l = self.split(latent)
        base_recon = self.base_decoder(torch.cat([b, s], dim=1))
        lora_recon = self.lora_decoder(torch.cat([s, l], dim=1))
        return torch.cat([base_recon, lora_recon], dim=1), latent

    def intervention_basis(self) -> torch.Tensor:
        basis = []
        base_w = self.base_decoder.weight.T
        lora_w = self.lora_decoder.weight.T
        for idx in range(self.base_exclusive_dim):
            basis.append(-base_w[idx])
        for idx in range(self.shared_dim):
            basis.append(lora_w[self.base_exclusive_dim + idx] - base_w[self.base_exclusive_dim + idx])
        for idx in range(self.lora_exclusive_dim):
            basis.append(lora_w[self.shared_dim + idx])
        return torch.stack(basis, dim=0)


class _DeltaAE(torch.nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared_dim = max(2, latent_dim // 3)
        self.delta_dim = latent_dim - self.shared_dim
        self.encoder = torch.nn.Linear(hidden_dim * 2, latent_dim)
        self.delta_decoder = torch.nn.Linear(latent_dim, hidden_dim, bias=False)
        self.mean_decoder = torch.nn.Linear(self.shared_dim, hidden_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        delta_recon = self.delta_decoder(latent)
        mean_recon = self.mean_decoder(latent[:, : self.shared_dim])
        return torch.cat([delta_recon, mean_recon], dim=1), latent

    def intervention_basis(self) -> torch.Tensor:
        return self.delta_decoder.weight.T


def _encode_relu(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, topk: int) -> np.ndarray:
    latent = np.maximum(0.0, features @ weight + bias)
    if topk > 0 and latent.size:
        idx = np.argpartition(latent, -min(topk, latent.shape[1]), axis=1)[:, -min(topk, latent.shape[1]) :]
        mask = np.zeros_like(latent)
        rows = np.arange(latent.shape[0])[:, None]
        mask[rows, idx] = 1.0
        latent = latent * mask
    return latent.astype(np.float32)


def _pair_source(base: np.ndarray, lora: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "delta":
        return lora - base
    if input_mode == "pair_concat":
        return np.concatenate([base, lora], axis=1)
    if input_mode == "delta_mean":
        return np.concatenate([lora - base, (lora + base) / 2.0], axis=1)
    if input_mode == "lora_only":
        return lora
    raise ValueError(f"Unknown input mode: {input_mode}")


def _rank_latents(features: np.ndarray, labels: np.ndarray) -> list[int]:
    if features.ndim == 1:
        return [0]
    centered_labels = labels.astype(np.float32) - labels.mean()
    scores = []
    for index in range(features.shape[1]):
        feat = features[:, index]
        centered_feat = feat - feat.mean()
        denom = float(np.linalg.norm(centered_feat) * np.linalg.norm(centered_labels))
        corr = 0.0 if denom <= EPS else float(np.dot(centered_feat, centered_labels) / denom)
        scores.append(abs(corr))
    return list(np.argsort(-np.asarray(scores)))


def _exclusivity_stats(base_feats: np.ndarray, lora_feats: np.ndarray) -> dict[str, float]:
    base_presence = np.abs(base_feats).mean(axis=0)
    lora_presence = np.abs(lora_feats).mean(axis=0)
    ratio = (lora_presence + EPS) / (base_presence + EPS)
    gap = lora_presence - base_presence
    return {
        "decoder_norm_ratio": float(ratio.mean()),
        "presence_gap_mean": float(gap.mean()),
        "shared_contamination": float(np.mean(np.abs(gap) < 0.05)),
    }


def _top_examples(bundle: dict[str, np.ndarray], features: np.ndarray, ranking: list[int], limit: int = 3) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if features.ndim == 1:
        features = features[:, None]
    for latent_id in ranking[: min(limit, features.shape[1])]:
        column = features[:, latent_id]
        top_idx = int(np.argmax(np.abs(column)))
        examples.append(
            {
                "latent_id": int(latent_id),
                "sample_id": str(bundle["sample_ids"][top_idx]),
                "topic": str(bundle["topics"][top_idx]),
                "template": str(bundle["templates"][top_idx]),
                "activation": float(column[top_idx]),
            }
        )
    return examples


def _load_behavior_eval_from_lineage(lineage: list[dict[str, Any]]) -> dict[str, Any] | None:
    for manifest in lineage:
        if manifest["stage"] == "train_lora" and "behavior_eval" in manifest["artifacts"]:
            return _read_json(Path(manifest["artifacts"]["behavior_eval"]))
    return None


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
