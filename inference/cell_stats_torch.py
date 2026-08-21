"""GPU-native counterparts of warm_dfps_manager's N-scale NumPy helpers
(_sqdist, _cell_stats, _median_nn_spacing, _local_nn_distance).

These are the ones that actually cost time: `_cell_stats` builds a full
N x M squared-distance matrix and reduces it twice (occupancy, faith/snap).
On CPU/NumPy, the column-wise (axis=0) reduction is cache-unfriendly and
dominates a warm frame's cost; on GPU there's no such penalty and no
sequential dependency to work around, so plain tensor ops suffice -- no
custom kernel needed here, unlike fps_refill's greedy loop.

Formulas mirror the NumPy versions exactly (same squared-distance
convention throughout, sqrt applied only once at the end where the NumPy
version does) -- small CPU-vs-GPU floating point drift is expected and
acceptable, per the "functionally equivalent" fidelity bar established (and
empirically validated against detection quality) in the fps_with_preidx
port.

Unlike the NumPy versions, these are unchunked: N x M at typical sizes
(e.g. 16384 x 4096, ~268MB float32) fits GPU memory comfortably, so a
single call avoids both the Python chunking loop and any reduction penalty.
"""
from __future__ import annotations

import torch


def _sqdist_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """All-pairs squared distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b."""
    d2 = (A * A).sum(dim=1)[:, None] + (B * B).sum(dim=1)[None, :] - 2.0 * (A @ B.T)
    return d2.clamp_(min=0.0)


def _cell_stats_torch(P: torch.Tensor, S: torch.Tensor):
    """Same contract as warm_dfps_manager._cell_stats: occupancy[m], faith[m],
    snap_idx[m] for samples S against cloud P. Single unchunked N x M pass."""
    P = P.contiguous().float()
    S = S.contiguous().float()
    M = S.shape[0]

    d2 = _sqdist_torch(P, S)  # (N, M)
    occupancy = torch.bincount(d2.argmin(dim=1), minlength=M)
    col_min, col_argmin = d2.min(dim=0)
    faith = torch.sqrt(col_min)
    snap_idx = col_argmin

    return occupancy, faith, snap_idx


def _median_nn_spacing_torch(S: torch.Tensor) -> float:
    """Flat, global length scale used when range_adaptive=False."""
    if S.shape[0] < 2:
        return 0.0
    S = S.contiguous().float()
    d2 = _sqdist_torch(S, S)
    d2.fill_diagonal_(float("inf"))
    return float(torch.median(torch.sqrt(d2.min(dim=1).values)))


def _local_nn_distance_torch(P: torch.Tensor, query_idx: torch.Tensor) -> torch.Tensor:
    """For each query point (indices into P), its nearest-neighbor distance
    to the rest of P."""
    P = P.contiguous().float()
    Q = P[query_idx]
    d2 = _sqdist_torch(Q, P)  # (len(query_idx), N)
    d2[torch.arange(Q.shape[0], device=P.device), query_idx] = float("inf")
    return torch.sqrt(d2.min(dim=1).values)
