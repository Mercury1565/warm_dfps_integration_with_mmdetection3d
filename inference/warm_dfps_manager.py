from __future__ import annotations
from dataclasses import dataclass
from email.policy import default
import numpy as np


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    All-pairs squared distances via ||a-b||² = ||a||² + ||b||² − 2a·b.
    """
    d2 = (A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2

def _cell_stats(P: np.ndarray, S: np.ndarray, chunk: int = 4096):
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
        d2 = _sqdist(P[lo : lo + chunk], S)
        occupancy += np.bincount(d2.argmin(axis=1), minlength=M)

        col_min = d2.min(axis=0)
        closer = col_min < faith_sq
        faith_sq[closer] = col_min[closer]
        snap_idx[closer] = d2.argmin(axis=0)[closer] + lo

    return occupancy, np.sqrt(faith_sq), snap_idx


def _median_nn_spacing(S: np.ndarray) -> float:
    """
    Median nearest-neighbour distance among the samples — the flat, global
    length scale used when range_adaptive=False.
    """
    if S.shape[0] < 2:
        return 0.0
    S = np.ascontiguousarray(S, dtype=np.float32)
    d2 = _sqdist(S, S)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def _local_nn_distance(P: np.ndarray, query_idx: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
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
        d2 = _sqdist(Q[lo:hi], P)
        d2[np.arange(hi - lo), idx[lo:hi]] = np.inf  # exclude each query's own point
        out[lo:hi] = np.sqrt(d2.min(axis=1))
    return out


def _thin_redundant(S: np.ndarray, keep: np.ndarray, occupancy: np.ndarray, sep_sq: np.ndarray) -> np.ndarray:
    """
    Greedily drop the lesser of any too-close pair of kept samples.
    ``sep_sq[i]`` is sample i's own (locally-adaptive) squared separation
    threshold.
    """
    S = np.ascontiguousarray(S, dtype=np.float32)
    d2 = _sqdist(S, S)
    np.fill_diagonal(d2, np.inf)
    for i in np.argsort(occupancy, kind="stable"):
        if not keep[i]:
            continue
        close = (d2[i] < sep_sq[i]) & keep
        close[i] = False
        if close.any():
            keep[i] = False
    return keep


def fps_refill(P: np.ndarray, seed_idx: np.ndarray, num_samples: int, chunk: int = 4096) -> np.ndarray:
    """
    Greedy FPS over ``P`` continued from already-selected ``seed_idx``.

    NumPy reference of ``farthest_point_sample_with_preidx``: seeds the
    min-distance field with ``P[seed_idx]``, then repeatedly picks the cloud
    point farthest from everything selected so far. Returns ``num_samples``
    indices into ``P``, seeds first.
    """
    P = np.ascontiguousarray(P, dtype=np.float32)
    N = P.shape[0]
    seed_idx = np.asarray(seed_idx, dtype=np.int64).ravel()

    if seed_idx.size == 0:
        raise ValueError("fps_refill needs at least one seed (cold start feeds "
                         "one random index — see module docstring).")
    if num_samples > N:
        raise ValueError(f"num_samples ({num_samples}) must be <= N ({N}).")

    have = min(seed_idx.size, num_samples)
    out = np.empty(num_samples, dtype=np.int64)
    out[:have] = seed_idx[:have]
    if have == num_samples:
        return out

    seeds = P[out[:have]]
    min_dist = np.empty(N, dtype=np.float32)
    for lo in range(0, N, chunk):
        min_dist[lo : lo + chunk] = np.sqrt(
            _sqdist(P[lo : lo + chunk], seeds).min(axis=1))

    while have < num_samples:
        j = int(min_dist.argmax())
        out[have] = j
        np.minimum(min_dist, np.linalg.norm(P - P[j], axis=1), out=min_dist)
        have += 1

    return out


@dataclass
class StepResult:
    preidx: np.ndarray
    cold: bool
    reason: str
    n_carried: int
    n_kept: int
    n_dropped: int
    n_snap_merged: int
    median_spacing: float

    @property
    def kept_fraction(self) -> float:
        return self.n_kept / self.n_carried if self.n_carried else 0.0


class WarmStartManager:
    """
    Carries layer-1 D-FPS samples across frames of a LiDAR sequence.
    """

    def __init__(self, 
    num_samples: int, 
    min_occupancy: int = 1,
    stale_factor: float = 2.0, 
    separation_factor: float = 0.5,
    min_kept_fraction: float = 0.25, 
    seed_count: int | None = None,
    seed: int | None = None, 
    range_adaptive: bool = True
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if seed_count is not None and not (0 < seed_count < num_samples):
            raise ValueError("seed_count must lie in (0, num_samples)")
        self.num_samples = num_samples
        self.min_occupancy = min_occupancy
        self.stale_factor = stale_factor
        self.separation_factor = separation_factor
        self.min_kept_fraction = min_kept_fraction
        self.seed_count = seed_count
        self.range_adaptive = range_adaptive
        self._rng = np.random.default_rng(seed)
        self._S_prev: np.ndarray | None = None
        self._pending_reason = "first-frame"

    # ── state control ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget the carried samples (call at a sequence/drive boundary)."""
        self._S_prev = None
        self._pending_reason = "reset"

    @property
    def carrying(self) -> bool:
        return self._S_prev is not None

    # ── the per-frame decision ───────────────────────────────────────────────

    def step(self, P: np.ndarray, transform: np.ndarray | None = None) -> StepResult:
        P = np.asarray(P, dtype=np.float32)

        if P.ndim != 2:
            raise ValueError(f"P must be (N, dims), got {P.shape}")
        
        if P.shape[0] < self.num_samples:
            raise ValueError(f"cloud has {P.shape[0]} points < num_samples "
                             f"({self.num_samples})")

        # If we have no carried samples, this frame is a cold start
        if self._S_prev is None:
            return self._cold(P, self._pending_reason)

        # If we have carried samples, this frame is a warm start
        S = self._S_prev

        # If a transform is provided, apply it to the carried samples to bring them into the current frame's coordinates
        if transform is not None:
            d = S.shape[1]  # homogeneous (d+1, d+1); dim-agnostic like the engine
            R = np.asarray(transform[:d, :d], dtype=np.float32)
            t = np.asarray(transform[:d, d], dtype=np.float32)
            S = S @ R.T + t

        # Compute the occupancy, faith, and snap indices for the carried samples in the current point cloud
        occupancy, faith, snap_idx = _cell_stats(P, S)

        # Compute the local scale for each sample, either adaptively based on nearest 
        # neighbor distances or using a flat global median spacing
        if self.range_adaptive:
            uniq_snap, inverse = np.unique(snap_idx, return_inverse=True)
            local_scale = _local_nn_distance(P, uniq_snap)[inverse]  # per-sample
        else:
            local_scale = np.full(S.shape[0], _median_nn_spacing(S), dtype=np.float32)  # flat, global

        """
        Three independent gates, each optional via its factor being 0:
            - [occupancy >= min_occupancy] : a sample whose Voronoi cell is empty (nobody nearby claims it) is dead.
            - [faith <= stale_factor * local_scale] : a sample that drifted farther from any real point than stale_factor local-spacings 
                                                      is stale (default stale_factor=2.0: allow up to ~2 point-spacings of drift).
            - [_thin_redundant(...)] : declump survivors that are now too close together (separation_factor * local_scale threshold, 
                                       default half a point-spacing).
        """

        keep = occupancy >= self.min_occupancy
        if self.stale_factor > 0:
            keep &= faith <= self.stale_factor * local_scale
        if self.separation_factor > 0 and S.shape[0] > 1:
            sep_sq = (self.separation_factor * local_scale) ** 2
            keep = _thin_redundant(S, keep, occupancy, sep_sq)

        n_kept = int(keep.sum())
        n_carried = S.shape[0]

        # If fewer than min_kept_fraction (default 25%) of carried samples survived 
        # the gates above, warm-starting from the survivors isn't worth it, DO IT COLD!!!
        if n_carried and n_kept / n_carried < self.min_kept_fraction:
            return self._cold(P, "budget-fallback", n_carried=n_carried, n_kept=n_kept)

        # Two survivors can share a nearest new-frame point; duplicate seeds
        # would waste sample budget, so merge them and let the refill recover.
        if self.seed_count is None:
            preidx = np.unique(snap_idx[keep])
            n_merged = n_kept - preidx.size
        else:
            # Fixed-K mode: rank survivors by cell occupancy (most load-bearing
            # first), dedupe snap collisions keeping the best-ranked, cut to K.

            # First rankd survivors by occupancy descending
            order = np.argsort(-occupancy[keep], kind="stable")

            # reorder snap_idx by that rank & take the first occurrence of each unique value in that ranked order
            snapped = snap_idx[keep][order]
            _, first_pos = np.unique(snapped, return_index=True)

            # restore the rank ordering so the best-ranked duplicate is the one kept, not an arbitrary one            
            uniq = snapped[np.sort(first_pos)]
            n_merged = n_kept - uniq.size

            # if dedup drops below K, there's no way to hand back exactly K real seeds, so it's forced cold
            if uniq.size < self.seed_count:
                return self._cold(P, "seed-deficit", n_carried=n_carried, n_kept=n_kept)

            # otherwise, truncate to K
            preidx = uniq[: self.seed_count]

        return StepResult(
            preidx=preidx, cold=False, reason="warm",
            n_carried=n_carried, n_kept=n_kept, n_dropped=n_carried - n_kept,
            n_snap_merged=n_merged, median_spacing=float(np.median(local_scale)),
        )

    def commit(self, S_new: np.ndarray) -> None:
        """
        Store this frame's final sample positions ((M, dims)) as next frame's
        carry. 
        """

        S_new = np.asarray(S_new, dtype=np.float32)
        if S_new.ndim != 2 or S_new.shape[0] != self.num_samples:
            raise ValueError(f"expected ({self.num_samples}, dims) sample "
                             f"positions, got {S_new.shape}")
        
        self._S_prev = S_new.copy()

    def _cold(self, P: np.ndarray, reason: str, n_carried: int = 0, n_kept: int = 0) -> StepResult:
        """
        The uniform fallback for every "give up on warm-start" path
        """
        first = np.array([self._rng.integers(P.shape[0])], dtype=np.int64)
        return StepResult(
            preidx=first, cold=True, reason=reason,
            n_carried=n_carried, n_kept=n_kept,
            n_dropped=n_carried - n_kept, n_snap_merged=0,
            median_spacing=0.0,
        )

    # ── standalone mode (no TF): step + reference refill + commit ────────────

    def resample(self, P: np.ndarray, transform: np.ndarray | None = None):
        """
        Convenience wrapper for the common case of "step, refill, commit" in one go. 
        Returns the new sample positions, their indices into P, and the StepResult for 
        this frame.
        """
        res = self.step(P, transform=transform)
        idx = fps_refill(P, res.preidx, self.num_samples)
        S = np.asarray(P, dtype=np.float32)[idx]
        self.commit(S)
        return S, idx, res
