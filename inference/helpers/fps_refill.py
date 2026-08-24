import numpy as np

from .warm_dfps_helpers import sqdist

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
            sqdist(P[lo : lo + chunk], seeds).min(axis=1))

    while have < num_samples:
        j = int(min_dist.argmax())
        out[have] = j
        np.minimum(min_dist, np.linalg.norm(P - P[j], axis=1), out=min_dist)
        have += 1

    return out