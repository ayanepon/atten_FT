# -*- coding: utf-8 -*-
"""Mixed-precision helpers for sample-wise additional training."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Optional, Tuple

import torch


def amp_enabled(requested: Optional[bool] = None) -> bool:
    if requested is not None:
        return bool(requested) and torch.cuda.is_available()
    env = os.environ.get("EXTRACT_AMP", "1").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return torch.cuda.is_available()


def autocast_context(enabled: Optional[bool] = None):
    """bf16 autocast on CUDA when available; no-op otherwise."""
    if not amp_enabled(enabled):
        return nullcontext()
    # Prefer bf16 (no GradScaler needed) when supported
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def maybe_scaler(enabled: Optional[bool] = None) -> Any:
    """GradScaler only for fp16 autocast (not needed for bf16)."""
    if not amp_enabled(enabled):
        return None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return None
    try:
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        return None


def enable_tf32() -> None:
    """Allow TF32 matmul/cudnn on Ampere+ (safe speed win)."""
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
