from __future__ import annotations
from dataclasses import dataclass
from inference.helpers.warm_dfps_helpers import sqdist, cell_stats, local_nn_distance, median_nn_spacing
from inference.helpers.thin_redundant import thin_redundant
from inference.helpers.fps_refill import fps_refill
import numpy as np

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
            return self._cold(P.shape[0], self._pending_reason)

        # If we have carried samples, this frame is a warm start
        S = self._S_prev

        # If a transform is provided, apply it to the carried samples to bring them into the current frame's coordinates
        if transform is not None:
            d = S.shape[1]  # homogeneous (d+1, d+1); dim-agnostic like the engine
            R = np.asarray(transform[:d, :d], dtype=np.float32)
            t = np.asarray(transform[:d, d], dtype=np.float32)
            S = S @ R.T + t

        # Compute the occupancy, faith, and snap indices for the carried samples in the current point cloud
        occupancy, faith, snap_idx = cell_stats(P, S)

        # Compute the local scale for each sample, either adaptively based on nearest
        # neighbor distances or using a flat global median spacing
        if self.range_adaptive:
            uniq_snap, inverse = np.unique(snap_idx, return_inverse=True)
            local_scale = local_nn_distance(P, uniq_snap)[inverse]  # per-sample
        else:
            local_scale = np.full(S.shape[0], median_nn_spacing(S), dtype=np.float32)  # flat, global

        keep = self._compute_keep(S, occupancy, faith, local_scale)
        return self._decide(P.shape[0], S, keep, occupancy, snap_idx, local_scale)

    def _compute_keep(self, S: np.ndarray, occupancy: np.ndarray, faith: np.ndarray,
                      local_scale: np.ndarray) -> np.ndarray:
        """NumPy path for the three survival gates. step_gpu() computes the
        GPU-tensor equivalent itself rather than calling this."""
        keep = occupancy >= self.min_occupancy
        if self.stale_factor > 0:
            keep &= faith <= self.stale_factor * local_scale
        if self.separation_factor > 0 and S.shape[0] > 1:
            sep_sq = (self.separation_factor * local_scale) ** 2
            keep = thin_redundant(S, keep, occupancy, sep_sq)
        return keep

    def step_gpu(self, points, transform: np.ndarray | None = None) -> StepResult:
        """
        Torch/GPU-resident counterpart of step(): identical decision logic,
        but occupancy/faith/snap_idx/local_scale and the keep mask (incl.
        declumping, via thin_redundant_gpu) are computed on-device, so the
        full point cloud never has to leave the GPU -- only the small,
        M-sized results are pulled back to NumPy for _decide()'s tail.

        points: (N, dims) float32 CUDA tensor.
        """
        import torch
        from inference.helpers.warm_dfps_helpers_gpu import (cell_stats_torch, local_nn_distance_torch, median_nn_spacing_torch)
        from inference.helpers.thin_redundant_gpu import thin_redundant_gpu

        if points.ndim != 2:
            raise ValueError(f"points must be (N, dims), got {tuple(points.shape)}")
        
        if points.shape[0] < self.num_samples:
            raise ValueError(f"cloud has {points.shape[0]} points < num_samples "
                             f"({self.num_samples})")

        if self._S_prev is None:
            return self._cold(points.shape[0], self._pending_reason)

        if isinstance(self._S_prev, torch.Tensor):
            S = self._S_prev.to(device=points.device, dtype=torch.float32)
        else:
            S = torch.as_tensor(self._S_prev, device=points.device, dtype=torch.float32)

        if transform is not None:
            d = S.shape[1]
            R = torch.as_tensor(np.asarray(transform[:d, :d], dtype=np.float32),
                                device=points.device)
            t = torch.as_tensor(np.asarray(transform[:d, d], dtype=np.float32),
                                device=points.device)
            S = S @ R.T + t

        occupancy, faith, snap_idx = cell_stats_torch(points, S)

        if self.range_adaptive:
            uniq_snap, inverse = torch.unique(snap_idx, return_inverse=True)
            local_scale = local_nn_distance_torch(points, uniq_snap)[inverse]
        else:
            local_scale = torch.full((S.shape[0],), median_nn_spacing_torch(S),
                                     dtype=torch.float32, device=points.device)

        keep = occupancy >= self.min_occupancy
        if self.stale_factor > 0:
            keep = keep & (faith <= self.stale_factor * local_scale)
        if self.separation_factor > 0 and S.shape[0] > 1:
            sep_sq = (self.separation_factor * local_scale) ** 2
            keep = thin_redundant_gpu(S, keep, occupancy, sep_sq)

        return self._decide_gpu(points.shape[0], S, keep, occupancy, snap_idx, local_scale)

    def _decide(self, n_points: int, S: np.ndarray, keep: np.ndarray,
               occupancy: np.ndarray, snap_idx: np.ndarray,
               local_scale: np.ndarray) -> StepResult:
        """
        Shared tail of step()/step_gpu(): given the already-decided keep
        mask (from _compute_keep() on CPU, or the GPU-tensor equivalent in
        step_gpu()), finish the decision and assemble the StepResult.
        """
        n_kept = int(keep.sum())
        n_carried = S.shape[0]

        # If fewer than min_kept_fraction (default 25%) of carried samples survived 
        # the gates above, warm-starting from the survivors isn't worth it, DO IT COLD!!!
        if n_carried and n_kept / n_carried < self.min_kept_fraction:
            return self._cold(n_points, "budget-fallback", n_carried=n_carried, n_kept=n_kept)

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
                return self._cold(n_points, "seed-deficit", n_carried=n_carried, n_kept=n_kept)

            # otherwise, truncate to K
            preidx = uniq[: self.seed_count]

        return StepResult(
            preidx=preidx, cold=False, reason="warm",
            n_carried=n_carried, n_kept=n_kept, n_dropped=n_carried - n_kept,
            n_snap_merged=n_merged, median_spacing=float(np.median(local_scale)),
        )

    def _decide_gpu(self, n_points: int, S, keep, occupancy, snap_idx,
                    local_scale) -> StepResult:
        """
        Torch-tensor counterpart of _decide(): stays GPU-resident
        throughout, including `preidx` in the returned StepResult -- only
        small Python scalars (n_kept, median_spacing, ...) cross back to
        CPU, never arrays.
        """
        import torch

        n_kept = int(keep.sum().item())
        n_carried = S.shape[0]

        if n_carried and n_kept / n_carried < self.min_kept_fraction:
            return self._cold(n_points, "budget-fallback", n_carried=n_carried, n_kept=n_kept)

        if self.seed_count is None:
            preidx = torch.unique(snap_idx[keep])
            n_merged = n_kept - preidx.numel()
        else:
            order = torch.argsort(-occupancy[keep], stable=True)
            snapped = snap_idx[keep][order]

            # First occurrence (by rank position) of each unique value:
            # stable-sort by value (ties keep relative rank order), take
            # each run's first element's original position, then restore
            # rank order -- torch.unique has no return_index equivalent.
            sorted_snapped, sort_idx = torch.sort(snapped, stable=True)
            is_first = torch.ones_like(sorted_snapped, dtype=torch.bool)
            is_first[1:] = sorted_snapped[1:] != sorted_snapped[:-1]
            first_pos = torch.sort(sort_idx[is_first]).values
            uniq = snapped[first_pos]
            n_merged = n_kept - uniq.numel()

            if uniq.numel() < self.seed_count:
                return self._cold(n_points, "seed-deficit", n_carried=n_carried, n_kept=n_kept)
            preidx = uniq[: self.seed_count]

        return StepResult(
            preidx=preidx, cold=False, reason="warm",
            n_carried=n_carried, n_kept=n_kept, n_dropped=n_carried - n_kept,
            n_snap_merged=n_merged,
            median_spacing=float(torch.quantile(local_scale, 0.5).item()),
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

    def commit_gpu(self, S_new) -> None:
        """GPU-resident counterpart of commit(): stores S_new (a CUDA
        tensor) directly, no NumPy round-trip -- so step_gpu() doesn't need
        to re-upload it next frame."""
        if S_new.ndim != 2 or S_new.shape[0] != self.num_samples:
            raise ValueError(f"expected ({self.num_samples}, dims) sample "
                             f"positions, got {tuple(S_new.shape)}")
        self._S_prev = S_new.detach().clone()

    def _cold(self, n_points: int, reason: str, n_carried: int = 0, n_kept: int = 0) -> StepResult:
        """
        The uniform fallback for every "give up on warm-start" path
        """
        first = np.array([self._rng.integers(n_points)], dtype=np.int64)
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
