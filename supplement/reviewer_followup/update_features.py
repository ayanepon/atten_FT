"""Gradient and LoRA parameter-update features for matched baselines."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
import torch


LAYER_PATTERNS = (
    re.compile(r"(?:gpt_neox\.layers|transformer\.h|model\.layers)\.(\d+)"),
    re.compile(r"(?:layers|h)\.(\d+)"),
)


def parameter_scope(name: str) -> str:
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return f"layer_{int(match.group(1)):02d}"
    prefix = name.split(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9_]+", "_", prefix) or "other"


def _scopes(name: str) -> tuple[str, str]:
    return parameter_scope(name), "global"


def _empty_accumulator() -> MutableMapping[str, float]:
    return defaultdict(float)


def _finalize_gradient(acc: Mapping[str, float]) -> Dict[str, float]:
    grad_l2 = math.sqrt(max(0.0, acc.get("grad_l2_sq", 0.0)))
    weight_l2 = math.sqrt(max(0.0, acc.get("weight_l2_sq", 0.0)))
    denom = max(grad_l2 * weight_l2, 1e-30)
    return {
        "n_parameters": int(acc.get("n_parameters", 0.0)),
        "grad_l1": float(acc.get("grad_l1", 0.0)),
        "grad_l2": grad_l2,
        "grad_max": float(acc.get("grad_max", 0.0)),
        "weight_l2_before": weight_l2,
        "grad_weight_cosine": float(acc.get("grad_weight_dot", 0.0) / denom),
    }

def _finalize_delta(acc: Mapping[str, float]) -> Dict[str, float]:
    delta_l2 = math.sqrt(max(0.0, acc.get("delta_l2_sq", 0.0)))
    weight_l2 = math.sqrt(max(0.0, acc.get("weight_l2_sq", 0.0)))
    grad_l2 = math.sqrt(max(0.0, acc.get("grad_l2_sq", 0.0)))
    return {
        "n_parameters": int(acc.get("n_parameters", 0.0)),
        "delta_l1": float(acc.get("delta_l1", 0.0)),
        "delta_l2": delta_l2,
        "delta_max": float(acc.get("delta_max", 0.0)),
        "update_to_weight_ratio": float(delta_l2 / max(weight_l2, 1e-30)),
        "grad_delta_cosine": float(acc.get("grad_delta_dot", 0.0) / max(grad_l2 * delta_l2, 1e-30)),
    }


def initial_gradient_features(
    model,
    train_enc: Mapping[str, torch.Tensor],
    *,
    model_device_fn,
    move_batch_fn,
) -> Tuple[List[Dict[str, float]], Dict[str, torch.Tensor]]:
    """Measure pre-update gradients without changing model parameters."""
    model.eval()
    model.zero_grad(set_to_none=True)
    batch = move_batch_fn(dict(train_enc), model_device_fn(model))
    outputs = model(**batch, output_attentions=False, use_cache=False)
    outputs.loss.backward()
    accumulators: Dict[str, MutableMapping[str, float]] = defaultdict(_empty_accumulator)
    gradients: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        weight = parameter.detach().float()
        gradients[name] = grad.cpu().clone()
        for scope in _scopes(name):
            acc = accumulators[scope]
            acc["n_parameters"] += grad.numel()
            acc["grad_l1"] += float(grad.abs().sum().item())
            acc["grad_l2_sq"] += float(grad.square().sum().item())
            acc["grad_max"] = max(acc["grad_max"], float(grad.abs().max().item()))
            acc["weight_l2_sq"] += float(weight.square().sum().item())
            acc["grad_weight_dot"] += float((grad * weight).sum().item())
    model.zero_grad(set_to_none=True)
    rows = [{"feature_family": "gradient", "scope": scope, **_finalize_gradient(acc)} for scope, acc in sorted(accumulators.items())]
    return rows, gradients


def parameter_delta_features(
    model,
    before_state: Mapping[str, torch.Tensor],
    initial_gradients: Mapping[str, torch.Tensor],
) -> List[Dict[str, float]]:
    accumulators: Dict[str, MutableMapping[str, float]] = defaultdict(_empty_accumulator)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in before_state:
            continue
        before = before_state[name].detach().to(device=parameter.device, dtype=torch.float32)
        after = parameter.detach().float()
        delta = after - before
        grad = initial_gradients.get(name)
        grad_device = torch.zeros_like(delta) if grad is None else grad.to(device=delta.device, dtype=delta.dtype)
        for scope in _scopes(name):
            acc = accumulators[scope]
            acc["n_parameters"] += delta.numel()
            acc["delta_l1"] += float(delta.abs().sum().item())
            acc["delta_l2_sq"] += float(delta.square().sum().item())
            acc["delta_max"] = max(acc["delta_max"], float(delta.abs().max().item()))
            acc["weight_l2_sq"] += float(before.square().sum().item())
            acc["grad_l2_sq"] += float(grad_device.square().sum().item())
            acc["grad_delta_dot"] += float((grad_device * delta).sum().item())
    return [{"feature_family": "parameter_delta", "scope": scope, **_finalize_delta(acc)} for scope, acc in sorted(accumulators.items())]


def overfit_fixed_steps_with_gradient_curve(
    model,
    train_enc: Mapping[str, torch.Tensor],
    *,
    steps: int,
    lr: float,
    model_device_fn,
    move_batch_fn,
    use_amp: bool = True,
) -> Tuple[List[float], List[float], int, List[float]]:
    """Matched fixed-step update while recording the global gradient norm."""
    from hardsplit.amp_utils import autocast_context, maybe_scaler

    model.train()
    batch = move_batch_fn(dict(train_enc), model_device_fn(model))
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found")
    optimizer = torch.optim.AdamW(params, lr=lr)
    scaler = maybe_scaler(use_amp)
    losses: List[float] = []
    gradient_norms: List[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            outputs = model(**batch, output_attentions=False, use_cache=False)
            loss = outputs.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_sq = torch.zeros((), device=params[0].device, dtype=torch.float32)
        for parameter in params:
            if parameter.grad is not None:
                grad_sq = grad_sq + parameter.grad.detach().float().square().sum()
        losses.append(float(loss.detach().float().item()))
        gradient_norms.append(float(grad_sq.sqrt().item()))
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    model.eval()
    accuracies = [float("nan")] * len(losses)
    return losses, accuracies, int(steps), gradient_norms


def curve_summary(values: Iterable[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan"), f"{prefix}_slope": float("nan")}
    slope = 0.0 if len(arr) == 1 else float(np.polyfit(np.arange(len(arr), dtype=float), arr, 1)[0])
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        f"{prefix}_slope": slope,
    }
