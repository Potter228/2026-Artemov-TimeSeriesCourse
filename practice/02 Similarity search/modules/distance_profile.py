import numpy as np
from numba import njit
from modules.utils import z_normalize


@njit
def _brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool) -> np.ndarray:
    """Visit every window and every point: O((n-m+1) * m)."""
    m = len(query)
    profile = np.empty(len(ts) - m + 1)
    for start in range(len(profile)):
        mean, std = 0.0, 1.0
        if is_normalize:
            for j in range(m):
                mean += ts[start + j]
            mean /= m
            variance = 0.0
            for j in range(m):
                variance += (ts[start + j] - mean) ** 2
            std = np.sqrt(variance / m)
            if std == 0:
                # A constant window has no defined z-normalization.
                profile[start] = np.inf
                continue
        squared_distance = 0.0
        for j in range(m):
            difference = (ts[start + j] - mean) / std - query[j]
            squared_distance += difference ** 2
        profile[start] = np.sqrt(squared_distance)
    return profile


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """Compute all Euclidean distances, optionally normalizing each window."""
    ts = np.asarray(ts, dtype=float)
    query = np.asarray(query, dtype=float)
    if ts.ndim != 1 or query.ndim != 1 or not 0 < len(query) <= len(ts):
        raise ValueError("Expected 1D series with 0 < len(query) <= len(ts)")
    if not (np.isfinite(ts).all() and np.isfinite(query).all()):
        raise ValueError("Time series must contain finite values")
    if is_normalize:
        query = z_normalize(query)
    return _brute_force(ts, query, is_normalize)
