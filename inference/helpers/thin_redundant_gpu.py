"""
GPU-native counterpart of warm_dfps_manager._thin_redundant.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from .warm_dfps_helpers_gpu import _sqdist_torch

_CSRC = Path(__file__).parent.parent / "csrc"

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")

_ext = load(
    name="thin_redundant_ext",
    sources=[
        str(_CSRC / "thin_redundant.cpp"),
        str(_CSRC / "thin_redundant_kernel.cu"),
    ],
    verbose=True,
)


@torch.no_grad()
def thin_redundant_gpu(S: torch.Tensor, keep: torch.Tensor, occupancy: torch.Tensor,
                       sep_sq: torch.Tensor) -> torch.Tensor:
    """Drop-in CUDA counterpart of ``_thin_redundant(S, keep, occupancy, sep_sq)``.

    S          (M, 3) float32 CUDA tensor -- carried sample positions
    keep       (M,) bool CUDA tensor -- keep mask so far (occupancy/staleness gates already applied)
    occupancy  (M,) integer CUDA tensor -- used only to compute the fixed processing order
    sep_sq     (M,) float32 CUDA tensor -- per-sample squared separation threshold

    Returns an (M,) bool CUDA tensor: the thinned keep mask.
    """
    m = S.shape[0]
    if m <= 1:
        return keep.clone()

    S = S.contiguous()
    if S.dtype != torch.float32:
        S = S.float()

    d2 = _sqdist_torch(S, S)
    d2.fill_diagonal_(float("inf"))
    d2 = d2.contiguous()

    order = torch.argsort(occupancy, stable=True).contiguous()
    sep_sq = sep_sq.contiguous()
    if sep_sq.dtype != torch.float32:
        sep_sq = sep_sq.float()
    keep_i32 = keep.to(torch.int32).contiguous()

    _ext.thin_redundant_forward(d2, order, sep_sq, keep_i32, m)

    return keep_i32.to(torch.bool)
