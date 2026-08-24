import numpy as np

from .warm_dfps_helpers import sqdist

def thin_redundant(S: np.ndarray, keep: np.ndarray, occupancy: np.ndarray, sep_sq: np.ndarray) -> np.ndarray:
    """
    Greedily drop the lesser of any too-close pair of kept samples.
    ``sep_sq[i]`` is sample i's own (locally-adaptive) squared separation
    threshold.
    """
    S = np.ascontiguousarray(S, dtype=np.float32)
    d2 = sqdist(S, S)
    np.fill_diagonal(d2, np.inf)
    for i in np.argsort(occupancy, kind="stable"):
        if not keep[i]:
            continue
        close = (d2[i] < sep_sq[i]) & keep
        close[i] = False
        if close.any():
            keep[i] = False
    return keep