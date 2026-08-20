from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_CSRC = Path(__file__).parent / "csrc"

# The env's ambient `nvcc` (via PATH) can resolve to a different CUDA install
# than the one PyTorch itself was built against; pin CUDA_HOME explicitly
# rather than trust whatever's ambient, but don't override an explicit
# user setting.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")

_ext = load(
    name="fps_with_preidx_ext",
    sources=[
        str(_CSRC / "fps_with_preidx.cpp"),
        str(_CSRC / "fps_with_preidx_kernel.cu"),
    ],
    verbose=True,
)


@torch.no_grad()
def farthest_point_sample_with_preidx(points: torch.Tensor, preidx: torch.Tensor,
                                      num_samples: int) -> torch.Tensor:
    """Drop-in CUDA counterpart of ``fps_refill(P, seed_idx, num_samples)``.

    points       (N, 3) float32 CUDA tensor
    preidx       (S,) integer CUDA tensor, indices into ``points``, S >= 1
    num_samples  total budget; returns this many indices, seeds first

    Returns an (num_samples,) int64 CUDA tensor of indices into ``points``.
    """
    if preidx.numel() == 0:
        raise ValueError("farthest_point_sample_with_preidx needs at least "
                         "one seed (cold start feeds one random index -- "
                         "see warm_dfps_manager's module docstring).")
    if num_samples > points.shape[0]:
        raise ValueError(f"num_samples ({num_samples}) must be <= N "
                         f"({points.shape[0]}).")

    preidx = preidx.to(device=points.device, dtype=torch.int64).contiguous()
    have = min(preidx.numel(), num_samples)
    preidx = preidx[:have]
    if have == num_samples:
        return preidx.clone()

    points = points.contiguous()
    if points.dtype != torch.float32:
        points = points.float()
    n = points.shape[0]

    idxs = torch.empty(num_samples, dtype=torch.int32, device=points.device)
    temp = torch.full((n,), 1e10, dtype=torch.float32, device=points.device)

    _ext.farthest_point_sampling_with_preidx_forward(
        points, preidx, temp, idxs, n, num_samples, have)

    return idxs.to(torch.int64)
