from __future__ import annotations

import numpy as np

def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    All-pairs squared distances via ||a-b||² = ||a||² + ||b||² − 2a·b.
    """
    d2 = (A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2

def cell_stats(P: np.ndarray, S: np.ndarray, chunk: int = 4096):
    """
    One chunked sweep of point-to-sample distances, returning everything the
    classifier *and* the snap step need:

        occupancy[m]  — cloud points whose nearest sample is m (Voronoi count)
        faith[m]      — distance from sample m to its nearest cloud point
        snap_idx[m]   — index (into P) of that nearest cloud point
    """
    P = np.ascontiguousarray(P, dtype=np.float32)
    S = np.ascontiguousarray(S, dtype=np.float32)
    N, M = P.shape[0], S.shape[0]

    occupancy = np.zeros(M, dtype=np.int64)
    faith_sq = np.full(M, np.inf, dtype=np.float32)
    snap_idx = np.zeros(M, dtype=np.int64)

    for lo in range(0, N, chunk):
        d2 = sqdist(P[lo : lo + chunk], S)
        occupancy += np.bincount(d2.argmin(axis=1), minlength=M)

        col_min = d2.min(axis=0)
        closer = col_min < faith_sq
        faith_sq[closer] = col_min[closer]
        snap_idx[closer] = d2.argmin(axis=0)[closer] + lo

    return occupancy, np.sqrt(faith_sq), snap_idx


def median_nn_spacing(S: np.ndarray) -> float:
    """
    Median nearest-neighbour distance among the samples — the flat, global
    length scale used when range_adaptive=False.
    """
    if S.shape[0] < 2:
        return 0.0
    S = np.ascontiguousarray(S, dtype=np.float32)
    d2 = sqdist(S, S)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def local_nn_distance(P: np.ndarray, query_idx: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
    """
    For a chosen set of query points (query_idx) drawn from P, 
    computes each query's nearest-neighbor distance to the rest of P 
    """
    P = np.ascontiguousarray(P, dtype=np.float32)
    N = P.shape[0]
    idx = np.arange(N) if query_idx is None else np.ascontiguousarray(query_idx)
    Q = P[idx]
    out = np.empty(idx.shape[0], dtype=np.float32)
    for lo in range(0, idx.shape[0], chunk):
        hi = min(lo + chunk, idx.shape[0])
        d2 = sqdist(Q[lo:hi], P)
        d2[np.arange(hi - lo), idx[lo:hi]] = np.inf  # exclude each query's own point
        out[lo:hi] = np.sqrt(d2.min(axis=1))
    return out
