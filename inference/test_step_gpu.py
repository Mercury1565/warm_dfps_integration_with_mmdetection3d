"""Parity check for WarmStartManager.step_gpu() against the NumPy step()
it's meant to replace on the warm path. No test framework in this
directory -- plain asserts, run directly:

    python test_step_gpu.py
"""
from __future__ import annotations

import numpy as np
import torch

from warm_dfps_manager import WarmStartManager


def _results_match(a, b, tol: float = 0.02) -> list[str]:
    problems = []
    if a.cold != b.cold:
        problems.append(f"cold mismatch: {a.cold} vs {b.cold}")
    if a.reason != b.reason:
        problems.append(f"reason mismatch: {a.reason} vs {b.reason}")
    if a.n_carried != b.n_carried:
        problems.append(f"n_carried mismatch: {a.n_carried} vs {b.n_carried}")
    if abs(a.n_kept - b.n_kept) > max(2, int(0.01 * max(a.n_carried, 1))):
        problems.append(f"n_kept diverged too far: {a.n_kept} vs {b.n_kept}")
    if abs(a.median_spacing - b.median_spacing) > tol * max(1.0, abs(a.median_spacing)):
        problems.append(f"median_spacing mismatch: {a.median_spacing} vs {b.median_spacing}")
    return problems


def _run_case(rng: np.random.Generator, n: int, num_samples: int, n_frames: int,
             range_adaptive: bool, device: str) -> None:
    mgr_cpu = WarmStartManager(num_samples=num_samples, seed=0, range_adaptive=range_adaptive)
    mgr_gpu = WarmStartManager(num_samples=num_samples, seed=0, range_adaptive=range_adaptive)

    for frame in range(n_frames):
        P = rng.standard_normal((n, 3)).astype(np.float32)
        P_t = torch.from_numpy(P).to(device)

        res_cpu = mgr_cpu.step(P)
        res_gpu = mgr_gpu.step_gpu(P_t)

        problems = _results_match(res_cpu, res_gpu)
        assert not problems, (
            f"n={n} num_samples={num_samples} frame={frame} "
            f"range_adaptive={range_adaptive}: {problems}")

        # advance both managers identically: use step()'s own preidx -> full
        # sample set via ordinary FPS-style fill (just take the first
        # num_samples farthest-ish points deterministically -- any fill
        # works as long as both managers commit the SAME S, since the point
        # here is to check step()/step_gpu() agree given identical carried
        # state, not to re-validate fps_refill).
        idx = np.argsort(rng.random(n))[:num_samples]
        # guarantee the seeds are included so commit() reflects a plausible
        # continuation of this frame's decision
        idx = np.unique(np.concatenate([res_cpu.preidx, idx]))[:num_samples]
        if idx.size < num_samples:
            idx = np.arange(num_samples)
        S = P[idx]
        mgr_cpu.commit(S)
        mgr_gpu.commit(S)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required for this test"
    device = "cuda:0"
    rng = np.random.default_rng(0)

    cases = [
        (2000, 256, 6, True),
        (2000, 256, 6, False),
        (5000, 512, 5, True),
        (500, 128, 8, True),   # forces frequent budget-fallback/reset cycles
    ]
    for n, num_samples, n_frames, range_adaptive in cases:
        _run_case(rng, n, num_samples, n_frames, range_adaptive, device)
        print(f"OK  n={n} num_samples={num_samples} n_frames={n_frames} "
              f"range_adaptive={range_adaptive}")

    print("All cases passed.")


if __name__ == "__main__":
    main()
