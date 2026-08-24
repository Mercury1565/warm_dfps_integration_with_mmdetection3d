"""
GPU-native counterparts of warm_dfps_manager's N-scale NumPy helpers
(_sqdist, _cell_stats, _median_nn_spacing, _local_nn_distance).
"""
from __future__ import annotations
import torch


def sqdist_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """All-pairs squared distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b."""
    d2 = (A * A).sum(dim=1)[:, None] + (B * B).sum(dim=1)[None, :] - 2.0 * (A @ B.T)
    return d2.clamp_(min=0.0)


def cell_stats_torch(P: torch.Tensor, S: torch.Tensor):
    """Same contract as warm_dfps_manager._cell_stats: occupancy[m], faith[m],
    snap_idx[m] for samples S against cloud P. Single unchunked N x M pass."""
    P = P.contiguous().float()
    S = S.contiguous().float()
    M = S.shape[0]

    d2 = sqdist_torch(P, S)  # (N, M)
    occupancy = torch.bincount(d2.argmin(dim=1), minlength=M)
    col_min, col_argmin = d2.min(dim=0)
    faith = torch.sqrt(col_min)
    snap_idx = col_argmin

    return occupancy, faith, snap_idx


def median_nn_spacing_torch(S: torch.Tensor) -> float:
    """Flat, global length scale used when range_adaptive=False."""
    if S.shape[0] < 2:
        return 0.0
    S = S.contiguous().float()
    d2 = sqdist_torch(S, S)
    d2.fill_diagonal_(float("inf"))
    return float(torch.median(torch.sqrt(d2.min(dim=1).values)))


def local_nn_distance_torch(P: torch.Tensor, query_idx: torch.Tensor) -> torch.Tensor:
    """For each query point (indices into P), its nearest-neighbor distance
    to the rest of P."""
    P = P.contiguous().float()
    Q = P[query_idx]
    d2 = sqdist_torch(Q, P)  # (len(query_idx), N)
    d2[torch.arange(Q.shape[0], device=P.device), query_idx] = float("inf")
    return torch.sqrt(d2.min(dim=1).values)
