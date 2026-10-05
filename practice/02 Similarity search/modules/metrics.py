import numpy as np
from numba import njit


def _validate_pair(ts1: np.ndarray, ts2: np.ndarray) -> tuple:
    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)
    if ts1.ndim != 1 or ts1.shape != ts2.shape or ts1.size == 0:
        raise ValueError("Expected nonempty one-dimensional series of equal length")
    if not (np.isfinite(ts1).all() and np.isfinite(ts2).all()):
        raise ValueError("Time series must contain finite values")
    return ts1, ts2


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """Euclidean distance between two series of equal length."""
    ts1, ts2 = _validate_pair(ts1, ts2)
    return float(np.linalg.norm(ts1 - ts2))


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """Euclidean distance after separate z-normalization of both series."""
    ts1, ts2 = _validate_pair(ts1, ts2)
    std1, std2 = np.std(ts1), np.std(ts2)
    if std1 == 0 or std2 == 0:
        raise ValueError("Z-normalization is undefined for constant series")
    return ED_distance((ts1 - ts1.mean()) / std1,
                       (ts2 - ts2.mean()) / std2)


@njit
def _dtw_cost(ts1: np.ndarray, ts2: np.ndarray, radius: int) -> float:
    """Accumulate squared costs within the band, keeping only two DP rows."""
    n = len(ts1)
    previous = np.full(n + 1, np.inf)
    current = np.full(n + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        start = max(1, i - radius)
        stop = min(n, i + radius)
        # Clear the two boundaries so stale values cannot enter the band.
        current[start - 1] = np.inf
        if stop < n:
            current[stop + 1] = np.inf
        for j in range(start, stop + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous
    return previous[n]


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """DTW squared cost; r in [0, 1] is the relative Sakoe-Chiba radius.

    The integer radius is floor(r * n). At r=0 only the diagonal is used;
    at r=1 the path is unrestricted. No square root is taken, as in sktime.
    Numba compiles the same recurrence without changing the search algorithm.
    """
    ts1, ts2 = _validate_pair(ts1, ts2)
    if not np.isfinite(r) or not 0 <= r <= 1:
        raise ValueError("r must be between 0 and 1")
    return float(_dtw_cost(ts1, ts2, int(r * len(ts1))))
